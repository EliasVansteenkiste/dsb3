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
patch_size = (64, 64, 64)
p_transform = {'patch_size': patch_size,
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
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'][:7], train_valid_ids['valid'][:7]

train_data_iterator = data_iterators.PatchCentersPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           batch_size=chunk_size,
                                                                           transform_params=p_transform,
                                                                           data_prep_fun=data_prep_function_train,
                                                                           rng=rng,
                                                                           patient_ids=train_pids,
                                                                           full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.PatchCentersPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           batch_size=1,
                                                                           transform_params=p_transform,
                                                                           data_prep_fun=data_prep_function_valid,
                                                                           rng=rng,
                                                                           patient_ids=valid_pids,
                                                                           full_batch=False, random=False,
                                                                           infinite=False)

nchunks_per_epoch = 10
max_nchunks = 100000

validate_every = int(1. * nchunks_per_epoch)
save_every = int(4. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 3e-5,
    int(max_nchunks * 0.5): 2e-6,
    int(max_nchunks * 0.6): 1e-6,
    int(max_nchunks * 0.9): 1e-7
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

    l = conv3d(l_in, num_filters=64)
    l = conv3d(l, num_filters=64)
    l = conv3d(l, num_filters=64)

    l = max_pool3d(l)

    l = conv3d(l, num_filters=64)
    l = conv3d(l, num_filters=64)
    l = conv3d(l, num_filters=64)

    # Z
    l_dz = nn.layers.DenseLayer(l, num_units=64, W=nn.init.Orthogonal("relu"),
                                b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.very_leaky_rectify)
    mu_z = nn.layers.DenseLayer(l_dz, num_units=1, W=nn.init.Orthogonal(),
                                b=nn.init.Constant(patch_size[0] / 2), nonlinearity=nn.nonlinearities.softplus)
    sigma_z = nn.layers.DenseLayer(l_dz, num_units=1, W=nn.init.Orthogonal(),
                                   b=nn.init.Constant(1.), nonlinearity=nn.nonlinearities.softplus)

    lz = nn_lung.NormalCDFLayer(mu_z, sigma_z, patch_size[0])

    # Y
    l_dy = nn.layers.DenseLayer(l, num_units=64, W=nn.init.Orthogonal("relu"),
                                b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.very_leaky_rectify)
    mu_y = nn.layers.DenseLayer(l_dy, num_units=1, W=nn.init.Orthogonal(),
                                b=nn.init.Constant(patch_size[1] / 2), nonlinearity=nn.nonlinearities.softplus)
    sigma_y = nn.layers.DenseLayer(l_dy, num_units=1, W=nn.init.Orthogonal(),
                                   b=nn.init.Constant(1.), nonlinearity=nn.nonlinearities.softplus)

    ly = nn_lung.NormalCDFLayer(mu_y, sigma_y, patch_size[1])

    # X
    l_dx = nn.layers.DenseLayer(l, num_units=64, W=nn.init.Orthogonal("relu"),
                                b=nn.init.Constant(0.1), nonlinearity=nn.nonlinearities.very_leaky_rectify)
    mu_x = nn.layers.DenseLayer(l_dx, num_units=1, W=nn.init.Orthogonal(),
                                b=nn.init.Constant(patch_size[2] / 2), nonlinearity=nn.nonlinearities.softplus)
    sigma_x = nn.layers.DenseLayer(l_dx, num_units=1, W=nn.init.Orthogonal(),
                                   b=nn.init.Constant(1.), nonlinearity=nn.nonlinearities.softplus)

    lx = nn_lung.NormalCDFLayer(mu_x, sigma_x, patch_size[2])

    lzd = nn.layers.DimshuffleLayer(lz, (0, 'x', 1))
    lyd = nn.layers.DimshuffleLayer(ly, (0, 'x', 1))
    lxd = nn.layers.DimshuffleLayer(lx, (0, 'x', 1))

    l_mu = nn.layers.ConcatLayer([mu_z, mu_y, mu_x])

    l_out = nn.layers.ConcatLayer([lzd, lyd, lxd])

    return namedtuple('Model', ['l_in', 'l_out', 'l_target', 'lz', 'ly', 'lx', 'l_mu'])(l_in, l_out, l_target,
                                                                                        lz, ly, lx, l_mu)


def build_objective(model, deterministic=False):
    pred_z = nn.layers.get_output(model.lz, deterministic=deterministic)
    pred_y = nn.layers.get_output(model.ly, deterministic=deterministic)
    pred_x = nn.layers.get_output(model.lx, deterministic=deterministic)

    targets = nn.layers.get_output(model.l_target)
    tz_heaviside = nn_lung.heaviside(targets[:, :1], patch_size[0])
    ty_heaviside = nn_lung.heaviside(targets[:, 1:2], patch_size[1])
    tx_heaviside = nn_lung.heaviside(targets[:, 2:], patch_size[2])

    crps_z = T.mean((pred_z - tz_heaviside) ** 2)
    crps_y = T.mean((pred_y - ty_heaviside) ** 2)
    crps_x = T.mean((pred_x - tx_heaviside) ** 2)

    return crps_z + crps_y + crps_x


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
