import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn
import nn_lung
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import theano.tensor as T
import utils
import utils_lung
import os


restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (256, 256, 256)}

trans = 5
rot = 0
p_transform_augment = {
    'translation_range_z': [-trans, trans],
    'translation_range_y': [-trans, trans],
    'translation_range_x': [-trans, trans],
    'rotation_range_z': [-rot, rot],
    'rotation_range_y': [-rot, rot],
    'rotation_range_x': [-rot, rot]
}


def data_prep_function(data, p_transform, p_transform_augment):
    x = data_transforms.transform_dsb_segm(data=data,
                                     p_transform=p_transform,
                                     p_transform_augment=p_transform_augment)
    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment, p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

# data iterators
batch_size = 1

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)

train_data_iterator = data_iterators.DSBSegmDataGenerator(data_path=pathfinder.SEGM_PATH,
                                                            batch_size=batch_size,
                                                            transform_params=p_transform,
                                                            data_prep_fun=data_prep_function_train,
                                                            rng=rng,
                                                            patient_ids=train_pids,
                                                            random=True, infinite=True)

valid_data_iterator = data_iterators.DSBSegmDataGenerator(data_path=pathfinder.SEGM_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              data_prep_fun=data_prep_function_valid,
                                                              rng=rng,
                                                              patient_ids=valid_pids,
                                                              random=False, infinite=False)


test_data_iterator = data_iterators.DSBSegmDataGenerator(data_path=pathfinder.SEGM_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              data_prep_fun=data_prep_function_valid,
                                                              rng=rng,
                                                              patient_ids=test_pids,
                                                              random=False, infinite=False)


nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 30

validate_every = int(2 * nchunks_per_epoch)
save_every = int(1 * nchunks_per_epoch)

lr = 1e-5
learning_rate_schedule = {
    0: lr,
    int(5 * nchunks_per_epoch): lr/3.,
    int(10 * nchunks_per_epoch): lr/(3.)**2,
    int(25 * nchunks_per_epoch): lr/(3.)**3,
    int(28 * nchunks_per_epoch): lr/(3.)**4
}
for key in learning_rate_schedule: learning_rate_schedule[key] *= batch_size

# model

conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal("relu"),
                 nonlinearity=nn.nonlinearities.rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

nin = partial(nn.layers.NINLayer,
    W=nn.init.Orthogonal("relu"),
    b=nn.init.Constant(0.0),
    nonlinearity=nn.nonlinearities.rectify)

drop = nn.layers.DropoutLayer

dense = partial(nn.layers.DenseLayer,
                W=nn.init.Orthogonal("relu"),
                nonlinearity=nn.nonlinearities.rectify)


def build_model():
    l_in = nn.layers.InputLayer((None,) + p_transform['patch_size'])

    l = nn.layers.DimshuffleLayer(l_in, pattern=(0, 'x', 1, 2, 3))
    l = nn.layers.ExpressionLayer(l, lambda x: x/100.)

    l_target = nn.layers.InputLayer((None,))

    n = 32
    l = conv3d(l, n, filter_size=5, stride=2)

    n *= 2
    l = conv3d(l, n, filter_size=5, stride=2)

    n *= 2
    l = conv3d(l, n)
    l = conv3d(l, n)
    l = max_pool3d(l)

    n *= 2
    l = conv3d(l, n)
    l = conv3d(l, n)
    l = max_pool3d(l)

    # n *= 2
    l = conv3d(l, n)
    l = conv3d(l, n)
    l = max_pool3d(l)

    l = nin(l, num_units=64)

    # n *= 2
    l = dense(l, n)
    l = dense(l, n)

    l = nn.layers.DenseLayer(l,
                              num_units=1,
                              W=nn.init.Orthogonal(),#nn.init.Constant(0.0),
                              # b=nn.init.Constant(-np.log(1./0.25-1.)),
                              nonlinearity=nn.nonlinearities.sigmoid)
    l_out = nn.layers.ReshapeLayer(l, shape=(-1,))

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    p = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    p = T.clip(p, epsilon, 1.-epsilon)
    bce = T.nnet.binary_crossentropy(p, targets)
    return T.mean(bce)


def build_updates(train_loss, model, learning_rate):
    params = nn.layers.get_all_params(model.l_out, trainable=True)
    updates = nn.updates.adam(train_loss, params, learning_rate)
    return updates