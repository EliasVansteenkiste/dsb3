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
import nn_lung

restart_from_save = None
rng = np.random.RandomState(42)
p_transform = {'patch_size': (128, 128, 128),
               'mm_patch_size': (360, 400, 400),
               'pixel_spacing': (1., 1., 1.)
               }

p_transform_augment = {
    'translation_range_z': [-10, 10],
    'translation_range_y': [-10, 10],
    'translation_range_x': [-10, 10],
    'rotation_range_z': [-10, 10],
    'rotation_range_y': [-5, 5],
    'rotation_range_x': [-5, 5]
}

batch_size = 1
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk


def data_prep_function_train(data, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, annotations = data_transforms.transform_scan3d(data=x, p_transform=p_transform,
                                                      p_transform_augment=p_transform_augment, **kwargs)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations, shape='sphere')
    return x, y


def data_prep_function_test(data, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, annotations = data_transforms.transform_scan3d(data=x, p_transform=p_transform,
                                                      p_transform_augment=None, **kwargs)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations, shape='sphere')
    return x, y


train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)

train_data_iterator = data_iterators.LunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                       batch_size=chunk_size,
                                                       transform_params=p_transform,
                                                       data_prep_fun=data_prep_function_train,
                                                       rng=rng,
                                                       patient_ids=train_valid_ids['train'],
                                                       full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.LunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                       batch_size=chunk_size,
                                                       transform_params=p_transform,
                                                       data_prep_fun=data_prep_function_test,
                                                       rng=rng,
                                                       patient_ids=train_valid_ids['valid'],
                                                       full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 100
learning_rate_schedule = {
    0: 0.0002,
    int(max_nchunks * 0.1): 0.0001,
    int(max_nchunks * 0.3): 0.000075,
    int(max_nchunks * 0.6): 0.00005,
    int(max_nchunks * 0.9): 0.00001
}
validate_every = 2 * nchunks_per_epoch
save_every = 2 * nchunks_per_epoch

conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal('relu'),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)


def build_model():
    l_in = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])

    net = {}
    base_n_filters = 8
    net['contr_1_1'] = conv3d(l_in, base_n_filters)
    net['contr_1_2'] = conv3d(net['contr_1_1'], base_n_filters)
    net['pool1'] = max_pool3d(net['contr_1_2'])

    net['contr_2_1'] = conv3d(net['pool1'], base_n_filters * 2)
    net['contr_2_2'] = conv3d(net['contr_2_1'], base_n_filters * 2)
    net['pool2'] = max_pool3d(net['contr_2_2'])

    net['contr_3_1'] = conv3d(net['pool2'], base_n_filters * 4)
    net['contr_3_2'] = conv3d(net['contr_3_1'], base_n_filters * 4)
    net['pool3'] = max_pool3d(net['contr_3_2'])

    net['contr_4_1'] = conv3d(net['pool3'], base_n_filters * 8)
    net['contr_4_2'] = conv3d(net['contr_4_1'], base_n_filters * 8)
    l = net['pool4'] = max_pool3d(net['contr_4_2'])

    net['encode_1'] = conv3d(l, base_n_filters * 16)
    net['encode_2'] = conv3d(net['encode_1'], base_n_filters * 16)
    net['upscale1'] = nn_lung.Upscale3DLayer(net['encode_2'], 2)

    net['concat1'] = nn.layers.ConcatLayer([net['upscale1'], net['contr_4_2']],
                                           cropping=(None, None, "center", "center", "center"))
    net['expand_1_1'] = conv3d(net['concat1'], base_n_filters * 8)
    net['expand_1_2'] = conv3d(net['expand_1_1'], base_n_filters * 8)
    net['upscale2'] = nn_lung.Upscale3DLayer(net['expand_1_2'], 2)

    net['concat2'] = nn.layers.ConcatLayer([net['upscale2'], net['contr_3_2']],
                                           cropping=(None, None, "center", "center", "center"))
    net['expand_2_1'] = conv3d(net['concat2'], base_n_filters * 4)
    net['expand_2_2'] = conv3d(net['expand_2_1'], base_n_filters * 4)
    net['upscale3'] = nn_lung.Upscale3DLayer(net['expand_2_2'], 2)

    net['concat3'] = nn.layers.ConcatLayer([net['upscale3'], net['contr_2_2']],
                                           cropping=(None, None, "center", "center", "center"))
    net['expand_3_1'] = conv3d(net['concat3'], base_n_filters * 2)
    net['expand_3_2'] = conv3d(net['expand_3_1'], base_n_filters * 2)
    net['upscale4'] = nn_lung.Upscale3DLayer(net['expand_3_2'], 2)

    net['concat4'] = nn.layers.ConcatLayer([net['upscale4'], net['contr_1_2']],
                                           cropping=(None, None, "center", "center", "center"))
    net['expand_4_1'] = conv3d(net['concat4'], base_n_filters)
    net['expand_4_2'] = conv3d(net['expand_4_1'], base_n_filters)

    net['output_segmentation'] = dnn.Conv3DDNNLayer(net['expand_4_2'], num_filters=1,
                                                    filter_size=1,
                                                    W=nn.init.Orthogonal(),
                                                    nonlinearity=nn.nonlinearities.sigmoid)
    l_out = net['output_segmentation']

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-15):
    predictions = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    predictions = T.clip(predictions, epsilon, 1. - epsilon)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    ce = -T.mean(1e6 * T.log(predictions) * targets + T.log(1 - predictions) * (1 - targets))
    return ce


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
