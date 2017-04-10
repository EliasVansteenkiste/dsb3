# like dsb_af25lme_mal2_s5_p8a1, but 12 candidates instead of 8 and interpolation order 0 instead of 1

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
candidates_config = 'dsb_c3_s5_p8a1'

restart_from_save = None
rng = np.random.RandomState(42)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
candidates_path = predictions_dir + '/%s' % candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

# transformations
p_transform = {'patch_size': (48, 48, 48),
               'mm_patch_size': (48, 48, 48),
               'pixel_spacing': (1., 1., 1.),
               'order': 0,
               }

p_transform_augment = {
    'translation_range_z': [-5, 5],
    'translation_range_y': [-5, 5],
    'translation_range_x': [-5, 5],
    'rotation_range_z': [-10, 10],
    'rotation_range_y': [-10, 10],
    'rotation_range_x': [-10, 10]
}
n_candidates_per_patient = 12


def data_prep_function(data, patch_centers, pixel_spacing, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.transform_dsb_candidates(data=data,
                                                 patch_centers=patch_centers,
                                                 p_transform=p_transform,
                                                 p_transform_augment=p_transform_augment,
                                                 pixel_spacing=pixel_spacing)
    x = data_transforms.hu2normHU_low_clip(x)
    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform)
data_prep_function_tta = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform)


cutoff_p_nodule = 0.75
def candidates_prep_function(all_candidates, n_selection=None):
    if n_selection:
        all_candidates = all_candidates[:n_selection]

    selected_candidates = [] 
    for candidate in all_candidates:
        if candidate[-1]<cutoff_p_nodule:
            selected_candidates.append([-1,-1,-1,-1])
        else:
            selected_candidates.append(candidate)

    return selected_candidates

# data iterators
batch_size = 1

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids, stage2_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test'], train_valid_ids['test_stage2']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)
print 'n test', len(test_pids)
all_pids = train_pids + valid_pids + test_pids


id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)
id2label_test = utils_lung.read_test_labels(pathfinder.TEST_LABELS_PATH)


id2label_all = id2label.copy()
id2label_all.update(id2label_test)


train_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=batch_size,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_train,
                                                              candidates_prep_fun = candidates_prep_function,
                                                              id2candidates_path=id2candidates_path,
                                                              id2label = id2label_all,
                                                              rng=rng,
                                                              patient_ids=all_pids,
                                                              random=True, infinite=True)

valid_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_valid,
                                                              candidates_prep_fun = candidates_prep_function,
                                                              id2candidates_path=id2candidates_path,
                                                              id2label = id2label_all,
                                                              rng=rng,
                                                              patient_ids=valid_pids,
                                                              random=False, infinite=False)


test_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_valid,
                                                              candidates_prep_fun = candidates_prep_function,
                                                              id2candidates_path=id2candidates_path,
                                                              id2label = id2label_all,
                                                              rng=rng,
                                                              patient_ids=stage2_pids,
                                                              random=False, infinite=False)


tta_batch_size = 8


tta_test_data_iterator = data_iterators.DSBPatientsDataGeneratorTTA(data_path=pathfinder.DATA_PATH,
                                                              transform_params=p_transform,
                                                              id2candidates_path=id2candidates_path,
                                                              id2label = id2label,
                                                              data_prep_fun=data_prep_function_tta,
                                                              candidates_prep_fun = candidates_prep_function,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              patient_ids=test_pids,
                                                              tta = 64)


tta_valid_data_iterator = data_iterators.DSBPatientsDataGeneratorTTA(data_path=pathfinder.DATA_PATH,
                                                              transform_params=p_transform,
                                                              id2candidates_path=id2candidates_path,
                                                              id2label = id2label_test,
                                                              data_prep_fun=data_prep_function_tta,
                                                              candidates_prep_fun = candidates_prep_function,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              patient_ids=valid_pids,
                                                              tta = 64)



nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 10

validate_every = int(1 * nchunks_per_epoch)
save_every = int(0.25 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-5,
    int(5 * nchunks_per_epoch): 2e-6,
    int(6 * nchunks_per_epoch): 1e-6,
    int(7 * nchunks_per_epoch): 5e-7,
    int(9 * nchunks_per_epoch): 2e-7
}

# model
# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

drop = nn.layers.DropoutLayer

bn = nn.layers.batch_norm

dense = partial(nn.layers.DenseLayer,
                W=nn.init.Orthogonal('relu'),
                b=nn.init.Constant(0.0),
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

def load_pretrained_model(l_in):

    l = conv3d(l_in, 64)
    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2_red(l)

    l = dense(drop(l), 512)

    l = nn.layers.DenseLayer(l,1,nonlinearity=nn.nonlinearities.sigmoid, W=nn.init.Orthogonal(),
                b=nn.init.Constant(0))


    metadata = utils.load_pkl(os.path.join("/home/eavsteen/dsb3/storage/metadata/dsb3/models/","r_fred_malignancy_7-20170404-163552.pkl"))
    nn.layers.set_all_param_values(l, metadata['param_values'])

    return l


def build_model():
    l_in = nn.layers.InputLayer((None, n_candidates_per_patient,) + p_transform['patch_size'])
    l_in_rshp = nn.layers.ReshapeLayer(l_in, (-1, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((batch_size,))

    l = load_pretrained_model(l_in_rshp)

    #ins = penultimate_layer.output_shape[1]
    # l = conv3d(penultimate_layer, ins, filter_size=3, stride=2)
    # #l = feat_red(l)
    #
    #
    # l = nn.layers.DropoutLayer(l)
    # #
    # l = nn.layers.DenseLayer(l, num_units=256, W=nn.init.Orthogonal(),
    #                          nonlinearity=nn.nonlinearities.rectify)

    #l = nn.layers.DropoutLayer(l)

    l = nn.layers.ReshapeLayer(l, (-1, n_candidates_per_patient, 1))

    l_out = nn_lung.LogMeanExp(l,r=16, axis=(1, 2), name='LME')

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    p = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    p = T.clip(p, epsilon, 1.-epsilon)
    bce = T.nnet.binary_crossentropy(p, targets)
    return T.mean(bce)


def build_updates(train_loss, model, learning_rate):

    return nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
