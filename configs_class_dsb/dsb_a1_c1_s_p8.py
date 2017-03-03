import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import lasagne
import theano.tensor as T
import utils
# TODO: import correct config here
import configs_fpred_scan.dsb_c1_s_p8 as candidates_config

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (48, 48, 48),
               'mm_patch_size': (48, 48, 48),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-4, 4],
    'translation_range_y': [-4, 4],
    'translation_range_x': [-4, 4],
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}


def data_prep_function(data, patch_centers, pixel_spacing, p_transform,
                       p_transform_augment, world_coord_system, **kwargs):
    x = data_transforms.transform_scan3d(data=data,
                                         luna_annotations=None,
                                         p_transform=p_transform,
                                         p_transform_augment=p_transform_augment,
                                         pixel_spacing=pixel_spacing,
                                         world_coord_system=world_coord_system)
    x = data_transforms.pixelnormHU(x)
    # TODO x has to be a (n_patches, z, y, x)
    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform, world_coord_system=True)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform, world_coord_system=True)

# data iterators
batch_size = 4
nbatches_chunk = 8
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.CandidatesLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                 batch_size=chunk_size,
                                                                 transform_params=p_transform,
                                                                 data_prep_fun=data_prep_function_train,
                                                                 rng=rng,
                                                                 patient_ids=train_pids,
                                                                 full_batch=True, random=True, infinite=True,
                                                                 positive_proportion=0.5)

valid_data_iterator = data_iterators.CandidatesLunaValidDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                      transform_params=p_transform,
                                                                      data_prep_fun=data_prep_function_valid,
                                                                      patient_ids=valid_pids)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 100

validate_every = int(5. * nchunks_per_epoch)
save_every = int(1. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-5,
    int(max_nchunks * 0.5): 5e-6,
    int(max_nchunks * 0.6): 2e-6,
    int(max_nchunks * 0.8): 1e-6,
    int(max_nchunks * 0.9): 5e-7
}

# model
conv3 = partial(dnn.Conv3DDNNLayer,
                filter_size=3,
                pad='valid',
                W=nn.init.Orthogonal(),
                b=nn.init.Constant(0.01),
                nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool = partial(dnn.MaxPool3DDNNLayer,
                   pool_size=2)

drop = lasagne.layers.DropoutLayer

dense = partial(lasagne.layers.DenseLayer,
                W=lasagne.init.Orthogonal(),
                b=lasagne.init.Constant(0.01),
                nonlinearity=lasagne.nonlinearities.very_leaky_rectify)


def build_model():
    l_in = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((None, 1))

    l = conv3(l_in, num_filters=128)
    l = conv3(l, num_filters=128)

    l = max_pool(l)

    l = conv3(l, num_filters=128)
    l = conv3(l, num_filters=128)

    l = max_pool(l)

    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)

    l = max_pool(l)

    l_d01 = nn.layers.DenseLayer(l, num_units=1024, W=nn.init.Orthogonal(),
                                 b=nn.init.Constant(0.01), nonlinearity=nn.nonlinearities.very_leaky_rectify)

    l_d02 = nn.layers.DenseLayer(nn.layers.dropout(l_d01), num_units=1024, W=nn.init.Orthogonal(),
                                 b=nn.init.Constant(0.01), nonlinearity=nn.nonlinearities.very_leaky_rectify)

    l_out = nn.layers.DenseLayer(l_d02, num_units=2,
                                 W=nn.init.Constant(0.),
                                 nonlinearity=nn.nonlinearities.softmax)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.cast(T.flatten(nn.layers.get_output(model.l_target)), 'int32')
    p = predictions[T.arange(predictions.shape[0]), targets]
    p = T.clip(p, epsilon, 1.)
    loss = T.mean(T.log(p))
    return -loss


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
