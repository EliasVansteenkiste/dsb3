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

# TODO: import correct config here
candidates_config = 'dsb_c3_s2_p8a1'

restart_from_save = None
rng = np.random.RandomState(42)

#predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
#candidates_path = predictions_dir + '/%s' % candidates_config
candidates_path = pathfinder.METADATA_PATH + 'model-predictions/ikorshun/' + candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

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
batch_size = 4

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['training'], train_valid_ids['validation']

train_data_iterator = data_iterators.DSBPatientsBatchBalancedDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=batch_size,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_train,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=train_pids,
                                                              random=True, infinite=True)


valid_data_iterator = data_iterators.DSBPatientsBatchBalancedDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=batch_size,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_valid,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=valid_pids,
                                                              random=False, infinite=False)

nchunks_per_epoch = 50  # train_data_iterator.nsamples / chunk_size
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
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

drop = lasagne.layers.DropoutLayer

bn = lasagne.layers.batch_norm

dense = partial(lasagne.layers.DenseLayer,
    W=lasagne.init.Orthogonal('relu'),
    b=lasagne.init.Constant(0.01),
    nonlinearity=lasagne.nonlinearities.very_leaky_rectify)


def inrn_v2(lin):

    n_base_filter = 32

    l1 = conv3d(lin, n_base_filter, filter_size=1)

    l2 = conv3d(lin, n_base_filter, filter_size=1)
    l2 = conv3d(l2, n_base_filter, filter_size=3)

    l3 = conv3d(lin, n_base_filter, filter_size=1)
    l3 = conv3d(l3, n_base_filter, filter_size=3)
    l3 = conv3d(l3, n_base_filter, filter_size=3)

    l = lasagne.layers.ConcatLayer([l1, l2, l3])

    l = conv3d(l, lin.output_shape[1], filter_size=1)

    l = lasagne.layers.ElemwiseSumLayer([l,lin])

    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)

    return l

def inrn_v2_red(lin):
    #We want to reduce our total volume /4

    den = 16
    nom2 = 4
    nom3 = 5
    nom4 = 7

    ins = lin.output_shape[1]

    l1 = max_pool3d(lin)

    l2 = conv3d(lin, ins//den*nom2, filter_size=3, stride=2)

    l3 = conv3d(lin, ins//den*nom2, filter_size=1)
    l3 = conv3d(l3, ins//den*nom3, filter_size=3, stride=2)

    l4 = conv3d(lin, ins//den*nom2, filter_size=1)
    l4 = conv3d(l4, ins//den*nom3, filter_size=3)
    l4 = conv3d(l4, ins//den*nom4, filter_size=3, stride=2)

    l = lasagne.layers.ConcatLayer([l1, l2, l3, l4])

    return l

def build_model():
    l_in = nn.layers.InputLayer((None, n_candidates_per_patient,) + p_transform['patch_size'])

    l_target = nn.layers.InputLayer((None, 1))

    l = nn.layers.ReshapeLayer(l_in, (-1, 1) + p_transform['patch_size'])

    l = conv3d(l, 64)
    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)
    
    l = inrn_v2_red(l)
    l = inrn_v2_red(l)

    l = dense(drop(l), 256)
    l = dense(l, 2, nonlinearity=nn.nonlinearities.softmax)

    l = nn.layers.ReshapeLayer(l, (-1, n_candidates_per_patient, 2))

    l = dense(drop(l), 64)

    l_out = dense(drop(l), 2, nonlinearity=nn.nonlinearities.softmax)

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
