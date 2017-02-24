import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import theano.tensor as T
import utils

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (64, 64, 64),
               'mm_patch_size': (64, 64, 64),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-16, 16],
    'translation_range_y': [-16, 16],
    'translation_range_x': [-16, 16],
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}

zmuv_mean, zmuv_std = 0.36, 0.31


# data preparation function
def data_prep_function(data, patch_center, luna_annotations, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, **kwargs):
    x, patch_annotation_tf, annotations_tf = data_transforms.transform_patch3d(data=data,
                                                                               luna_annotations=luna_annotations,
                                                                               patch_center=patch_center,
                                                                               p_transform=p_transform,
                                                                               p_transform_augment=p_transform_augment,
                                                                               pixel_spacing=pixel_spacing,
                                                                               luna_origin=luna_origin)
    x = data_transforms.hu2normHU(x)
    x = data_transforms.zmuv(x, zmuv_mean, zmuv_std)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment, p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

# data iterators
batch_size = 4
nbatches_chunk = 8
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.PatchPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                    batch_size=chunk_size,
                                                                    transform_params=p_transform,
                                                                    data_prep_fun=data_prep_function_train,
                                                                    rng=rng,
                                                                    patient_ids=train_pids,
                                                                    full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.ValidPatchPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                         transform_params=p_transform,
                                                                         data_prep_fun=data_prep_function_valid,
                                                                         patient_ids=valid_pids)
if zmuv_mean is None or zmuv_std is None:
    print 'estimating ZMUV parameters'
    x_big = None
    for i, (x, _, _) in zip(xrange(4), train_data_iterator.generate()):
        print i
        x_big = x if x_big is None else np.concatenate((x_big, x), axis=0)
    zmuv_mean = x_big.mean()
    zmuv_std = x_big.std()
    print 'mean:', zmuv_mean
    print 'std:', zmuv_std

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 30

validate_every = int(2. * nchunks_per_epoch)
save_every = int(0.5 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-5,
    int(max_nchunks * 0.4): 5e-6,
    int(max_nchunks * 0.5): 2e-6,
    int(max_nchunks * 0.8): 1e-6,
    int(max_nchunks * 0.9): 5e-7
}

# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 W=nn.init.Orthogonal('relu'),
                 b=nn.init.Constant(0.0),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)


def conv_leaky_layer(l_in, n_filters, filter_size=3, pad='valid'):
    l = conv3d(l_in, n_filters, filter_size=filter_size, pad=pad)
    return l


def build_model():
    l_in = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])

    net = {}
    base_n_filters = 64
    net['contr_1_1'] = conv_leaky_layer(l_in, base_n_filters)
    net['contr_1_2'] = conv_leaky_layer(net['contr_1_1'], base_n_filters)
    net['contr_1_3'] = conv_leaky_layer(net['contr_1_2'], base_n_filters)
    net['contr_1_4'] = conv_leaky_layer(net['contr_1_3'], 2 * base_n_filters)
    net['pool1'] = max_pool3d(net['contr_1_4'])

    net['encode_0_1'] = conv_leaky_layer(net['pool1'], 2 * base_n_filters)
    net['encode_0_2'] = conv_leaky_layer(net['encode_0_1'], 2 * base_n_filters)
    net['encode_0_3'] = conv_leaky_layer(net['encode_0_2'], 2 * base_n_filters)

    net['encode_1_1'] = conv_leaky_layer(net['pool1'], 2 * base_n_filters, filter_size=1)
    net['encode_1_2'] = conv_leaky_layer(net['encode_1_1'], 2 * base_n_filters, filter_size=3)

    net['encode_2_1'] = conv_leaky_layer(net['pool1'], 2 * base_n_filters, filter_size=1)

    net['encode'] = nn.layers.ConcatLayer([net['encode_0_3'], net['encode_1_2'], net['encode_2_1']],
                                          cropping=(None, None, "center", "center", "center"))

    net['upscale1'] = nn.layers.Upscale3DLayer(net['encode'], 2)
    net['concat1'] = nn.layers.ConcatLayer([net['upscale1'], net['contr_1_4']],
                                           cropping=(None, None, "center", "center", "center"))

    net['dropout_1'] = nn.layers.dropout_channels(net['concat1'])

    net['expand_1_1'] = conv_leaky_layer(net['dropout_1'], 2 * base_n_filters)
    net['expand_1_2'] = conv_leaky_layer(net['expand_1_1'], base_n_filters)
    net['expand_1_3'] = conv_leaky_layer(net['expand_1_2'], base_n_filters)
    net['expand_1_4'] = conv_leaky_layer(net['expand_1_3'], base_n_filters / 2)
    net['expand_1_5'] = conv_leaky_layer(net['expand_1_4'], base_n_filters / 2)
    net['expand_1_6'] = conv_leaky_layer(net['expand_1_5'], base_n_filters / 2)

    l_out = dnn.Conv3DDNNLayer(net['expand_1_6'], num_filters=1,
                               filter_size=1,
                               nonlinearity=nn.nonlinearities.sigmoid)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    network_predictions = nn.layers.get_output(model.l_out)
    target_values = nn.layers.get_output(model.l_target)
    network_predictions, target_values = nn.layers.merge.autocrop([network_predictions, target_values],
                                                                  [None, None, 'center', 'center', 'center'])
    y_true_f = target_values
    y_pred_f = network_predictions

    intersection = T.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + epsilon) / (T.sum(y_true_f) + T.sum(y_pred_f) + epsilon)
    return -1. * dice


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
