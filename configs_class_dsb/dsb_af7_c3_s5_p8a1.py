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
import utils_lung
from densenet_fast_3D import dense_block,transition
import nn_lung

# TODO: import correct config here
candidates_config = 'dsb_c3_s5_p8a1'

restart_from_save = None
rng = np.random.RandomState(42)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
candidates_path = predictions_dir + '/%s' % candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

# transformations
p_transform = {'patch_size': (74, 74, 74),
               'mm_patch_size': (74, 74, 74),
               'pixel_spacing': (1., 1., 1.)
               }
n_candidates_per_patient = 4


def data_prep_function(data, patch_centers, pixel_spacing, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.transform_dsb_candidates(data=data,
                                                 patch_centers=patch_centers,
                                                 p_transform=p_transform,
                                                 p_transform_augment=p_transform_augment,
                                                 pixel_spacing=pixel_spacing)
    x = data_transforms.pixelnormHU(x)
    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform)

# data iterators
batch_size = 1

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)
print 'n test', len(test_pids)

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
test_data_iterator = data_iterators.DSBPatientsDataGeneratorTest(data_path=pathfinder.DATA_PATH,
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
    int(5 * nchunks_per_epoch): 2e-6,
    int(6 * nchunks_per_epoch): 1e-6,
    int(7 * nchunks_per_epoch): 5e-7,
    int(9 * nchunks_per_epoch): 2e-7
}

# model
dropout = 0.
num_blocks = 3
depth = 10
growth_rate = 12
first_output = 16

def build_model():
    l_in = nn.layers.InputLayer((None, n_candidates_per_patient, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((batch_size,))


    l = nn.layers.ReshapeLayer(l_in, (-1, 1,) + p_transform['patch_size'])

    l = dnn.Conv3DDNNLayer(l, first_output, 3, pad='valid',
                          W=nn.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None, name='pre_conv')
    l = nn.layers.BatchNormLayer(l, name='pre_bn', beta=None, gamma=None)

    if dropout:
        l = nn.layers.DropoutLayer(l, dropout)

    n = (depth - 1) // num_blocks
    for b in range(num_blocks):
        l = dense_block(l, n - 1, growth_rate, dropout,
                              name_prefix='block%d' % (b + 1))
        if b < num_blocks - 1:
            l = transition(l, dropout,
                                 name_prefix='block%d_trs' % (b + 1))
    # post processing until prediction
    l = nn.layers.ScaleLayer(l, name='post_scale')
    l = nn.layers.BiasLayer(l, name='post_shift')
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nn.nonlinearities.rectify,
                                name='post_relu')
    #network = nn.layers.GlobalPoolLayer(network, name='post_pool')



    l = dnn.Conv3DDNNLayer(l, first_output, 5, pad='valid',
                                 W=nn.init.HeNormal(gain='relu'),
                                 b=None, nonlinearity=None, name='final_conv')

    l = nn.layers.BatchNormLayer(l, name='pre_bn', beta=None, gamma=None)

    l = nn.layers.NonlinearityLayer(l, nonlinearity=nn.nonlinearities.rectify,
                                name='post_relu')

    l = nn.layers.DropoutLayer(l)

    l = nn.layers.DenseLayer(l, num_units=1, W=nn.init.Orthogonal(),
                             nonlinearity=None)

    l = nn.layers.ReshapeLayer(l, (-1, n_candidates_per_patient, 1))
    l_out = nn_lung.AggAllBenignExp(l)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)

def build_objective(model, deterministic=False, epsilon=1e-12):
    p = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    p = T.clip(p, epsilon, 1.-epsilon)
    bce = T.nnet.binary_crossentropy(p, targets)
    return T.mean(bce)

def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates