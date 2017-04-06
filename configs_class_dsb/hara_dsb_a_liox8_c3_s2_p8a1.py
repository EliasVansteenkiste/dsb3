# fred with order 0

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


predictions_config = 'dsb_a_liox8_c3_s2_p8a1'

restart_from_save = None
rng = np.random.RandomState(42)

n_candidates_per_patient = 8

batch_size = 1

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
# train_pids, valid_pids, test_pids = train_valid_ids['validation'], train_valid_ids['test'], []
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)

train_data_iterator = data_iterators.HaralickDataGenerator( data_path=pathfinder.HARALICK_PATH,
                                                            config=predictions_config,
                                                            batch_size=batch_size,
                                                            n_candidates_per_patient=n_candidates_per_patient,
                                                            rng=rng,
                                                            patient_ids=train_pids,
                                                            random=True, infinite=True)

valid_data_iterator = data_iterators.HaralickDataGenerator(data_path=pathfinder.HARALICK_PATH,
                                                           config=predictions_config,
                                                              batch_size=1,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              rng=rng,
                                                              patient_ids=valid_pids,
                                                              random=False, infinite=False)


test_data_iterator = data_iterators.HaralickDataGenerator(data_path=pathfinder.HARALICK_PATH,
                                                          config=predictions_config,
                                                              batch_size=1,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              rng=rng,
                                                              patient_ids=test_pids,
                                                              random=False, infinite=False)


nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 20

validate_every = int(1 * nchunks_per_epoch)
save_every = int(1 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-4,
    # 1: 1e-4,
    # 2: 1e-4,
    # 5: 1e-5,
    # 10: 1e-6,
    # 20: 1e-7,
    # 40: 1e-8,
    # int(5 * nchunks_per_epoch): 2e-6,
    # int(6 * nchunks_per_epoch): 1e-6,
    # int(7 * nchunks_per_epoch): 5e-7,
    # int(9 * nchunks_per_epoch): 2e-7
}

# model

drop = nn.layers.DropoutLayer

bn = nn.layers.batch_norm

dense = partial(nn.layers.DenseLayer,
                W=nn.init.Orthogonal('relu'),
                b=nn.init.Constant(0.0),
                nonlinearity=nn.nonlinearities.rectify)


def build_model():
    l_in = nn.layers.InputLayer((None, n_candidates_per_patient*169 + 1))
    l_target = nn.layers.InputLayer((None,))

    l_pred = nn.layers.SliceLayer(l_in, slice(0,1))
    l_feat = nn.layers.SliceLayer(l_in, slice(1,None))
    l = l_feat

    # l = nn.layers.ReshapeLayer(l, (-1, 169))

    l = drop(dense(l, num_units=64))
    l = drop(dense(l, num_units=64))
    l = dense(l, num_units=1, nonlinearity=None,
              W=nn.init.Orthogonal(), b=nn.init.Constant(0))
    # l = nn.layers.ReshapeLayer(l, (-1, n_candidates_per_patient, 1))
    # l = nn_lung.LogMeanExp(l, r=16, axis=(1, 2), name='LME')
    # l = nn.layers.ReshapeLayer(l, (-1, 1))

    l_pred = nn.layers.ExpressionLayer(l_pred, lambda x: T.log(x/(1.-x)))

    l = nn.layers.ConcatLayer([l, l_pred])
    # l = nn.layers.ElemwiseSumLayer([l, l_pred])
    # l = nn.layers.ReshapeLayer(l, (-1, 1))
    # l_out = nn.layers.ExpressionLayer(l, lambda x: nn.nonlinearities.sigmoid(x))
    # l_out = nn.layers.ReshapeLayer(l_out, (-1, 1))

    l_out = dense(l, num_units=1, nonlinearity=nn.nonlinearities.sigmoid,
              W=np.array([[0.01],[0.99]], "float32"), b=nn.init.Constant(0))
              # W=nn.init.Orthogonal(), b=nn.init.Constant(0))
              #   W =nn.init.Constant(1), b = nn.init.Constant(0))

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    p = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    p = T.clip(p, epsilon, 1.-epsilon)
    bce = T.nnet.binary_crossentropy(p, targets)
    # return T.mean(bce)
    if deterministic:
        return T.mean(bce)
    else: return T.mean(bce)  #+ 0.001*nn.regularization.regularize_network_params(model.l_out, nn.regularization.l2)


def build_updates(train_loss, model, learning_rate):
    params = nn.layers.get_all_params(model.l_out, trainable=True)
    updates = nn.updates.adam(train_loss, params, learning_rate)
    # updates = nn.updates.nesterov_momentum(train_loss, params, learning_rate)
    return updates