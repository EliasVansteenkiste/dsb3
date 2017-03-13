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
import glob
import utils_lung
from itertools import chain

# TODO: import correct config here
candidates_config = 'dsb_c3_s2_p8a1'

restart_from_save = None
rng = np.random.RandomState(42)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
candidates_path = predictions_dir + '/%s' % candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

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
batch_size = 2

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['training'], train_valid_ids['validation']
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

valid_data_iterator2 = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                               batch_size=4,
                                                               transform_params=p_transform,
                                                               n_candidates_per_patient=n_candidates_per_patient,
                                                               data_prep_fun=data_prep_function_valid,
                                                               id2candidates_path=id2candidates_path,
                                                               rng=rng,
                                                               patient_ids=valid_pids,
                                                               random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 10

validate_every = int(0.5 * nchunks_per_epoch)
save_every = int(0.25 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 4e-6,
    int(5 * nchunks_per_epoch): 2e-6,
    int(6 * nchunks_per_epoch): 1e-6,
    int(7 * nchunks_per_epoch): 5e-7,
    int(9 * nchunks_per_epoch): 2e-7
}

# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='valid',
                 W=nn.init.Orthogonal('relu'),
                 b=nn.init.Constant(0.0),
                 nonlinearity=nn.nonlinearities.identity)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)


def conv_prelu_layer(l_in, n_filters):
    l = conv3d(l_in, n_filters)
    l = nn.layers.ParametricRectifierLayer(l)
    return l


def dense_prelu_layer(l_in, num_units):
    l = nn.layers.DenseLayer(l_in, num_units=num_units, W=nn.init.Orthogonal(),
                             nonlinearity=nn.nonlinearities.linear)
    l = nn.layers.ParametricRectifierLayer(l)
    return l


def build_model():
    l_in = nn.layers.InputLayer((None, n_candidates_per_patient, 1,) + p_transform['patch_size'])
    l_in_rshp = nn.layers.ReshapeLayer(l_in, (-1, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((batch_size,))

    base_n_filters = 128
    l = conv_prelu_layer(l_in_rshp, n_filters=base_n_filters)
    l = conv_prelu_layer(l, n_filters=base_n_filters)
    l = conv_prelu_layer(l, n_filters=base_n_filters)

    l = max_pool3d(l)

    l = conv_prelu_layer(l, n_filters=base_n_filters)
    l = conv_prelu_layer(l, n_filters=base_n_filters)
    l = conv_prelu_layer(l, n_filters=base_n_filters)
    l_enc = conv_prelu_layer(l, n_filters=base_n_filters)

    num_units_dense = 512
    l_d01 = dense_prelu_layer(l, num_units=512)
    l_d01 = nn.layers.ReshapeLayer(l_d01, (-1, n_candidates_per_patient, num_units_dense))
    l_d02 = dense_prelu_layer(l_d01, num_units=512)
    l_out = nn.layers.DenseLayer(l_d02, num_units=2,
                                 W=nn.init.Constant(0.),
                                 b=np.array([np.log((1397. - 362) / 1398), np.log(362. / 1397)], dtype='float32'),
                                 nonlinearity=nn.nonlinearities.softmax)

    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, 'luna_p8a1')
    metadata = utils.load_pkl(metadata_path)
    for p, pv in zip(nn.layers.get_all_params(l_enc), metadata['param_values']):
        if p.get_value().shape != pv.shape:
            raise ValueError("mismatch: parameter has shape %r but value to "
                             "set has shape %r" %
                             (p.get_value().shape, pv.shape))
        p.set_value(pv)

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
