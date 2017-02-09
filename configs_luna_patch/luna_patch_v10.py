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
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
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
    return x, patch_annotation_tf


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform)

# data iterators
batch_size = 1
nbatches_chunk = 4
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.PatchCentersPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           batch_size=chunk_size,
                                                                           transform_params=p_transform,
                                                                           data_prep_fun=data_prep_function_train,
                                                                           rng=rng,
                                                                           patient_ids=train_valid_ids['train'],
                                                                           full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.PatchCentersPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           batch_size=1,
                                                                           transform_params=p_transform,
                                                                           data_prep_fun=data_prep_function_valid,
                                                                           rng=rng,
                                                                           patient_ids=train_valid_ids['valid'],
                                                                           full_batch=False, random=False,
                                                                           infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 30

validate_every = int(1. * nchunks_per_epoch)
save_every = int(0.5 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 3e-4,
    int(max_nchunks * 0.5): 2e-4,
    int(max_nchunks * 0.6): 1e-4,
    int(max_nchunks * 0.9): 1e-5
}

# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal('relu'),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2, stride=2)


def build_model():
    l_in = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((None, 3))

    l = conv3d(l_in, num_filters=32)
    l = conv3d(l, num_filters=32)

    l = max_pool3d(l)

    l = conv3d(l, num_filters=32)
    l = conv3d(l, num_filters=32)

    l = max_pool3d(l)

    l = conv3d(l, num_filters=64)
    l = conv3d(l, num_filters=64)
    l = conv3d(l, num_filters=64)

    l = max_pool3d(l)

    l = conv3d(l, num_filters=128)
    l = conv3d(l, num_filters=128)
    l = conv3d(l, num_filters=128)

    l = max_pool3d(l)

    l = conv3d(l, num_filters=128)
    l = conv3d(l, num_filters=128)
    l = conv3d(l, num_filters=128)

    l = max_pool3d(l)

    l_z = nn.layers.DenseLayer(l, num_units=p_transform['patch_size'][0],
                               nonlinearity=nn.nonlinearities.softmax)
    l_y = nn.layers.DenseLayer(l, num_units=p_transform['patch_size'][1],
                               nonlinearity=nn.nonlinearities.softmax)
    l_x = nn.layers.DenseLayer(l, num_units=p_transform['patch_size'][2],
                               nonlinearity=nn.nonlinearities.softmax)

    l_z = nn.layers.DimshuffleLayer(l_z, (0, 'x', 1))
    l_y = nn.layers.DimshuffleLayer(l_y, (0, 'x', 1))
    l_x = nn.layers.DimshuffleLayer(l_x, (0, 'x', 1))

    l_out = nn.layers.ConcatLayer([l_z, l_y, l_x], axis=1)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)

    pred_z = T.cumsum(predictions[:, 0, :], axis=-1)
    pred_y = T.cumsum(predictions[:, 1, :], axis=-1)
    pred_x = T.cumsum(predictions[:, 2, :], axis=-1)

    targets = nn.layers.get_output(model.l_target)
    tz_heaviside = nn_lung.heaviside(targets[:, :1], p_transform['patch_size'][0])
    ty_heaviside = nn_lung.heaviside(targets[:, 1:2], p_transform['patch_size'][1])
    tx_heaviside = nn_lung.heaviside(targets[:, 2:], p_transform['patch_size'][2])

    crps_z = T.mean((pred_z - tz_heaviside) ** 2)
    crps_y = T.mean((pred_y - ty_heaviside) ** 2)
    crps_x = T.mean((pred_x - tx_heaviside) ** 2)

    return crps_z + crps_y + crps_x


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
