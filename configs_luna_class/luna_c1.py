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
p_transform = {'patch_size': (32, 32, 32),
               'mm_patch_size': (32, 32, 32),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-3, 3],
    'translation_range_y': [-3, 3],
    'translation_range_x': [-3, 3],
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}

zmuv_mean, zmuv_std = None, None


def data_prep_function(data, patch_center, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, **kwargs):
    x, patch_annotation_tf = data_transforms.transform_patch3d(data=data,
                                                               patch_center=patch_center,
                                                               p_transform=p_transform,
                                                               p_transform_augment=p_transform_augment,
                                                               pixel_spacing=pixel_spacing,
                                                               luna_origin=luna_origin)
    x = data_transforms.hu2normHU(x)
    x = data_transforms.zmuv(x, zmuv_mean, zmuv_std)
    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment, p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

# data iterators
batch_size = 32
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.CandidatesLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                 batch_size=chunk_size,
                                                                 transform_params=p_transform,
                                                                 data_prep_fun=data_prep_function_train,
                                                                 rng=rng,
                                                                 patient_ids=train_pids,
                                                                 positive_proportion=0.5,
                                                                 full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.CandidatesLunaValidDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                      transform_params=p_transform,
                                                                      data_prep_fun=data_prep_function_valid,
                                                                      patient_ids=valid_pids)

print 'estimating ZMUV parameters'
x_big = None
for i, (x, _, _) in zip(xrange(1), train_data_iterator.generate()):
    x_big = x if x_big is None else np.concatenate((x_big, x), axis=0)
zmuv_mean = x_big.mean()
zmuv_std = x_big.std()
print 'mean:', zmuv_mean
print 'std:', zmuv_std

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 30

validate_every = int(5. * nchunks_per_epoch)
save_every = int(1. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-4,
    int(max_nchunks * 0.1): 6e-5,
    int(max_nchunks * 0.2): 5e-5,
    int(max_nchunks * 0.4): 4e-5,
    int(max_nchunks * 0.6): 3e-5,
    int(max_nchunks * 0.8): 2e-5,
    int(max_nchunks * 0.9): 1e-6
}

# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=5,
                 pad='valid',
                 W=nn.init.Orthogonal(),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

drop = nn.layers.DropoutLayer

dense = partial(nn.layers.DenseLayer,
                W=nn.init.Orthogonal('relu'),
                b=nn.init.Constant(0.0),
                nonlinearity=nn.nonlinearities.very_leaky_rectify)


def build_model():
    l_in = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((None, 1))

    l = conv3d(l_in, 64)
    l = max_pool3d(l)
    l = conv3d(l, 64)
    l = dense(l, 128)
    l_out = nn.layers.DenseLayer(nn.layers.dropout(l), num_units=2,
                                 W=nn.init.Constant(0.),
                                 nonlinearity=nn.nonlinearities.softmax)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = nn.layers.get_output(model.l_out)
    targets = T.cast(T.flatten(nn.layers.get_output(model.l_target)), 'int32')
    p = predictions[T.arange(predictions.shape[0]), targets]
    p = T.clip(p, epsilon, 1.)
    loss = T.mean(T.log(p))
    return -loss


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
