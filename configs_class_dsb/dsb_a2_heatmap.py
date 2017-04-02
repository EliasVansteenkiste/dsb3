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
candidates_config = 'dsb_c4'

restart_from_save = None
rng = np.random.RandomState(42)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
heatmap_path = predictions_dir + '/%s' % candidates_config
id2heatmap_path = utils_lung.get_candidates_paths(heatmap_path)
p_transform = {'patch_size': (32, 54, 54,54)
               }

def data_prep_function(data, **kwargs):
    x = np.concatenate(data,axis=2)
    return x


data_prep_function_train = data_prep_function
data_prep_function_valid = data_prep_function

# data iterators
batch_size = 4

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)

train_data_iterator = data_iterators.DSBPatientsDataHeatmapGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=batch_size,
                                                              transform_params=p_transform,
                                                              data_prep_fun=data_prep_function_train,
                                                              id2heatmap_path=id2heatmap_path,
                                                              rng=rng,
                                                              patient_ids=train_pids,
                                                              random=True, infinite=True)

valid_data_iterator = data_iterators.DSBPatientsDataHeatmapGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              data_prep_fun=data_prep_function_valid,
                                                              id2heatmap_path=id2heatmap_path,
                                                              rng=rng,
                                                              patient_ids=valid_pids,
                                                              random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 10

validate_every = int(0.5 * nchunks_per_epoch)
save_every = int(0.25 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-4,
    int(5 * nchunks_per_epoch): 2e-5,
    int(6 * nchunks_per_epoch): 5e-6,
    int(7 * nchunks_per_epoch): 1e-6,
    int(9 * nchunks_per_epoch): 3e-7
}

# model

conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

drop = nn.layers.DropoutLayer

dense = partial(nn.layers.DenseLayer,
                W=nn.init.Orthogonal(),
                nonlinearity=nn.nonlinearities.very_leaky_rectify)


def inrn_v2(lin):
    n_base_filter = 32

    l1 = conv3d(lin, n_base_filter, filter_size=1)

    l2 = conv3d(lin, n_base_filter, filter_size=1)
    l2 = conv3d(l2, n_base_filter, filter_size=3)

    l3 = conv3d(lin, n_base_filter, filter_size=1)
    l3 = conv3d(l3, n_base_filter, filter_size=3)
    l3 = conv3d(l3, n_base_filter, filter_size=3)

    l = nn.layers.ConcatLayer([l1, l2, l3])

    l = conv3d(l, lin.output_shape[1], filter_size=1)

    l = nn.layers.ElemwiseSumLayer([l, lin])

    l = nn.layers.NonlinearityLayer(l, nonlinearity=nn.nonlinearities.rectify)

    return l


def inrn_v2_red(lin):
    # We want to reduce our total volume /4

    den = 16
    nom2 = 4
    nom3 = 5
    nom4 = 7

    ins = lin.output_shape[1]

    l1 = max_pool3d(lin)

    l2 = conv3d(lin, ins // den * nom2, filter_size=3, stride=2)

    l3 = conv3d(lin, ins // den * nom2, filter_size=1)
    l3 = conv3d(l3, ins // den * nom3, filter_size=3, stride=2)

    l4 = conv3d(lin, ins // den * nom2, filter_size=1)
    l4 = conv3d(l4, ins // den * nom3, filter_size=3)
    l4 = conv3d(l4, ins // den * nom4, filter_size=3, stride=2)

    l = nn.layers.ConcatLayer([l1, l2, l3, l4])

    return l


def feat_red(lin):
    # We want to reduce the feature maps by a factor of 2
    ins = lin.output_shape[1]
    l = conv3d(lin, ins // 2, filter_size=1)
    return l


def build_model():
    l_in = nn.layers.InputLayer((None, ) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((batch_size,))

    l = conv3d(l_in, 32 *2 , filter_size=3, stride=2)
    l = feat_red(l)

    l = nn.layers.DenseLayer(l, num_units=128, W=nn.init.Orthogonal(),
                             nonlinearity=nn.nonlinearities.rectify)

    l_out = nn.layers.DenseLayer(l, num_units=1, W=nn.init.Orthogonal(),
                             nonlinearity=nn.nonlinearities.sigmoid)



    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    p = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    p = T.clip(p, epsilon, 1.-epsilon)
    bce = T.nnet.binary_crossentropy(p, targets)
    return T.mean(bce)


def build_updates(train_loss, model, learning_rate):
    final_layer=nn.layers.get_all_layers(model.l_out)[-3]
    param_final=final_layer.get_params(trainable=True)
    final_layer=nn.layers.get_all_layers(model.l_out)[-4]
    param_final.extend(final_layer.get_params(trainable=True))

    updates = nn.updates.adam(train_loss, param_final, learning_rate)
    return updates