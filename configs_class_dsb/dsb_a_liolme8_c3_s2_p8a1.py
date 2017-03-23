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

# TODO: import correct config here
candidates_config = 'dsb_c3_s2_p8a1_ls_elias'

restart_from_save = None
rng = np.random.RandomState(42)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
candidates_path = predictions_dir + '/%s' % candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

# transformations
p_transform = {'patch_size': (48, 48, 48),
               'mm_patch_size': (48, 48, 48),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-5, 5],
    'translation_range_y': [-5, 5],
    'translation_range_x': [-5, 5],
    'rotation_range_z': [-10, 10],
    'rotation_range_y': [-10, 10],
    'rotation_range_x': [-10, 10]
}
n_candidates_per_patient = 8


def data_prep_function(data, patch_centers, pixel_spacing, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.transform_dsb_candidates(data=data,
                                                 patch_centers=patch_centers,
                                                 p_transform=p_transform,
                                                 p_transform_augment=p_transform_augment,
                                                 pixel_spacing=pixel_spacing)
    x = data_transforms.pixelnormHU(x)
    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform)

# data iterators
batch_size = 1

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)

train_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=batch_size,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_train,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=train_pids,
                                                              random=True, infinite=True)

valid_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_valid,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=valid_pids,
                                                              random=False, infinite=False)

test_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_valid,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=test_pids,
                                                              random=False, infinite=False)


nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 10

validate_every = int(0.5 * nchunks_per_epoch)
save_every = int(0.25 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-5,
    int(5 * nchunks_per_epoch): 3e-6,
    int(6 * nchunks_per_epoch): 1e-6,
    int(7 * nchunks_per_epoch): 3e-7,
    int(9 * nchunks_per_epoch): 1e-7
}
for key in learning_rate_schedule.keys(): learning_rate_schedule[key] *= batch_size


# model

conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal('relu'),
                 nonlinearity=nn.nonlinearities.rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

drop = nn.layers.DropoutLayer

dense = partial(nn.layers.DenseLayer,
                W=nn.init.Orthogonal('relu'),
                nonlinearity=nn.nonlinearities.rectify)

conv2d = partial(dnn.Conv2DDNNLayer,
    filter_size=3,
    pad='same',
    W=nn.init.Orthogonal('relu'),
    nonlinearity=nn.nonlinearities.rectify)

max_pool2d = partial(dnn.MaxPool2DDNNLayer, pool_size=2)



def build_model():
    l_in = nn.layers.InputLayer((None, n_candidates_per_patient, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((batch_size,))

    # l = nn.layers.DimshuffleLayer(l_in, pattern=(0, 1, 3, 2, 4, 5))
    # l = nn.layers.ReshapeLayer(l, (-1, 1,) + p_transform['patch_size'][-2:])
    l = nn.layers.ReshapeLayer(l_in, (-1, 1,) + p_transform['patch_size'])

    n = 32
    l = conv3d(l, n)
    l = conv3d(l, n)
    l = max_pool3d(l) # 24

    n = 64
    l = conv3d(l, n)
    l = conv3d(l, n)
    l = max_pool3d(l) # 12

    n = 128
    l = conv3d(l, n)
    l = conv3d(l, n)
    l = max_pool3d(l)  # 6

    # n_features = np.prod(l.output_shape[-3:])
    # l = nn.layers.ReshapeLayer(l, (-1, p_transform['patch_size'][0], n_features))
    # l = nn.layers.FeaturePoolLayer(l, p_transform['patch_size'][0], axis=1)

    n = 256
    l = drop(dense(l, n))
    l = drop(dense(l, n))

    l = nn.layers.DenseLayer(l, num_units=1, W=nn.init.Orthogonal(),
                             nonlinearity=nn.nonlinearities.sigmoid, name='dense_p_benign')

    l = nn.layers.ReshapeLayer(l, (-1, n_candidates_per_patient, 1), name='reshape2patients')

    l_out = nn_lung.LogMeanExp(l, r=8, axis=(1, 2), name='LME')

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