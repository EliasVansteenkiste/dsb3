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

# transformations
p_transform = {'patch_size': (64, 64, 64),
               'mm_patch_size': (64, 64, 64),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-27, 27],
    'translation_range_y': [-27, 27],
    'translation_range_x': [-27, 27],
    'rotation_range_z': [-27, 27],
    'rotation_range_y': [-27, 27],
    'rotation_range_x': [-27, 27]
}


# data preparation function
def data_prep_function(data, patch_center, luna_annotations, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, patch_annotation_tf, annotations_tf = data_transforms.transform_patch3d(data=x,
                                                                               luna_annotations=luna_annotations,
                                                                               patch_center=patch_center,
                                                                               p_transform=p_transform,
                                                                               p_transform_augment=p_transform_augment,
                                                                               pixel_spacing=pixel_spacing,
                                                                               luna_origin=luna_origin)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment, p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

# data iterators
batch_size = 1
nbatches_chunk = 4
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

valid_data_iterator = data_iterators.PatchPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                    batch_size=1,
                                                                    transform_params=p_transform,
                                                                    data_prep_fun=data_prep_function_valid,
                                                                    rng=rng,
                                                                    patient_ids=valid_pids,
                                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 30

validate_every = int(1. * nchunks_per_epoch)
save_every = int(0.5 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-5,
    int(max_nchunks * 0.4): 5e-6,
    int(max_nchunks * 0.5): 3e-6,
    int(max_nchunks * 0.6): 2e-6,
    int(max_nchunks * 0.85): 1e-6,
    int(max_nchunks * 0.95): 5e-7
}

# model
n_resolutions = 3
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.linear)


def conv_block(l_in, num_filters, filter_size, stride):
    l_conv = conv3d(l_in, num_filters=num_filters, filter_size=filter_size, stride=stride)
    l_out = nn.layers.ParametricRectifierLayer(l_conv)
    return l_out


def join_block(l_in1, l_in2):
    l_in1_up = nn.layers.Upscale3DLayer(l_in1, 2)
    l_out = nn.layers.ConcatLayer([l_in1_up, l_in2], axis=1)
    return l_out


def build_model():
    model = {}
    model['input_0'] = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])

    resolutions = [str(k) for k in range(n_resolutions)[::-1]]
    for k in resolutions:
        model['input_r' + k] = dnn.Pool3DDNNLayer(model['input_0'], pool_size=2 ** int(k), mode='average_exc_pad')
        model['conv1_r' + k] = conv_block(model['input_r' + k], 32, 3, 1)
        model['conv2_r' + k] = conv_block(model['conv1_r' + k], 32, 3, 1)
        model['conv3_r' + k] = conv_block(model['conv2_r' + k], 32, 3, 1)

    model['conv33_r' + resolutions[0]] = model['conv3_r' + resolutions[0]]
    n_filters = [32 * i for i in range(2, n_resolutions + 1)]
    for k, f in zip(resolutions[1:], n_filters):
        model['join_r' + k] = join_block(model['conv33_r' + str(int(k) + 1)], model['conv3_r' + k])
        model['conv11_r' + k] = conv_block(model['join_r' + k], f, 3, 1)
        model['conv22_r' + k] = conv_block(model['conv11_r' + k], f, 3, 1)
        model['conv33_r' + k] = conv_block(model['conv22_r' + k], f, 3, 1)

    l_out = dnn.Conv3DDNNLayer(model['conv33_r0'], num_filters=1,
                               filter_size=1,
                               W=nn.init.Constant(0.),
                               nonlinearity=nn.nonlinearities.sigmoid)
    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(model['input_0'], l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = T.flatten(nn.layers.get_output(model.l_out))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    dice = (2. * T.sum(targets * predictions) + epsilon) / (T.sum(predictions) + T.sum(targets) + epsilon)
    return -1. * dice


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
