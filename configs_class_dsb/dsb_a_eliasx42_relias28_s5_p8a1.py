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

from configs_luna_props_patch import r_elias_28 as cfg_prop

# TODO: import correct config here
candidates_config = 'dsb_relias28_s5_p8a1'
###change here

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
               'heatmap_size': (48, 48, 48),
               'heatmap_order': 0,
               'heatmap_norm': 32.,
               'max_shape': (400, 400, 400)
               }


p_transform_augment = {
    'translation_range_z': [-2, 2],
    'translation_range_y': [-2, 2],
    'translation_range_x': [-2, 2],
    'rotation_range_z': [-10, 10],
    'rotation_range_y': [-10, 10],
    'rotation_range_x': [-10, 10]
}
n_candidates_per_patient = 8


def data_prep_function(data, candidates, pixel_spacing, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.build_dsb_can_heatmap(data=data,
                                                 candidates=candidates,
                                                 p_transform=p_transform,
                                                 p_transform_augment=p_transform_augment,
                                                 pixel_spacing=pixel_spacing)

    return x



def get_feature_dict(feature_vector):
    d_feat = {}
    unit_ptr = 0
    for obj_idx, obj_name in enumerate(cfg_prop.order_objectives):
        ptype = cfg_prop.property_type[obj_name]
        if ptype == 'classification':
            num_units = len(cfg_prop.property_bin_borders[obj_name])
            d_feat[obj_name] = feature_vector[unit_ptr:unit_ptr+num_units]
            unit_ptr += num_units
        elif ptype == 'continuous':
            d_feat[obj_name] = feature_vector[unit_ptr]
            unit_ptr += 1
        elif ptype == 'bounded_continuous':
            d_feat[obj_name] = feature_vector[unit_ptr]
            unit_ptr += 1
    return d_feat


def logodds2p(lo):
    if lo < -500:
        # prevent underflow
        return 0.
    elif lo > 500:
        # prevent overflow
        return 1.
    else:
        odds = np.exp(lo)
        p = odds / (1+odds)
        return p


def candidates_prep_function(all_candidates, n_selection=None):
    candidates_w_svalue = []
    for candidate in all_candidates:
        # print 'candidate', candidate
        feature_vector = candidate[4:]
        d_feat = get_feature_dict(feature_vector)
        # print d_feat
        sorting_value = d_feat['malignancy']
        can_w_svalue = np.concatenate((candidate[:3], [sorting_value]))  
        # print can_w_svalue
        candidates_w_svalue.append(can_w_svalue)

    a = np.asarray(sorted(candidates_w_svalue, key=lambda x: x[-1], reverse=True))
    
    if n_selection:
        a = a[:n_selection]

    return a



data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform)

# data iterators
batch_size = 12

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)
print 'n test', len(test_pids)

train_data_iterator = data_iterators.DSBPatientsDataGenerator_only_heatmap(data_path=pathfinder.DATA_PATH,
                                                              batch_size=batch_size,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_train,
                                                              candidates_prep_fun = candidates_prep_function,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=train_pids,
                                                              random=True, infinite=True)

valid_data_iterator = data_iterators.DSBPatientsDataGenerator_only_heatmap(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_valid,
                                                              candidates_prep_fun = candidates_prep_function,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=valid_pids,
                                                              random=False, infinite=False)


test_data_iterator = data_iterators.DSBPatientsDataGenerator_only_heatmap(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function_valid,
                                                              candidates_prep_fun = candidates_prep_function,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=test_pids,
                                                              random=False, infinite=False)


nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 8

validate_every = int(0.5 * nchunks_per_epoch)
save_every = int(0.5 * nchunks_per_epoch)


learning_rate_schedule = {
    0: 1e-4,
    int(5 * nchunks_per_epoch): 5e-5,
    int(6 * nchunks_per_epoch): 1e-5,
    int(7 * nchunks_per_epoch): 5e-6,
    int(9 * nchunks_per_epoch): 1e-6
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


def load_pretrained_model(l_in):

    l = conv3d(l_in, 64)
    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2_red(l)

    l = dense(drop(l), 512)

    d_final_layers = {}
    final_layers = []
    unit_ptr = 0

    for obj_idx, obj_name in enumerate(cfg_prop.order_objectives):
        ptype = cfg_prop.property_type[obj_name]

        if ptype == 'classification':
            num_units = len(cfg_prop.property_bin_borders[obj_name])
            l_fin = nn.layers.DenseLayer(l, num_units=num_units,
                     W=nn.init.Orthogonal(),
                     b=nn.init.Constant(cfg_prop.init_values_final_units[obj_name]),
                     nonlinearity=nn.nonlinearities.softmax, name='dense_softmax_'+ptype+'_'+obj_name)

        elif ptype == 'continuous':
            l_fin = nn.layers.DenseLayer(l, num_units=1,
                    W=nn.init.Orthogonal(),
                    b=nn.init.Constant(cfg_prop.init_values_final_units[obj_name]),
                    nonlinearity=nn.nonlinearities.softplus, name='dense_softplus_'+ptype+'_'+obj_name)

        elif ptype == 'bounded_continuous':
            l_fin = nn.layers.DenseLayer(l, num_units=1,
                    W=nn.init.Orthogonal(),
                    b=nn.init.Constant(cfg_prop.init_values_final_units[obj_name]),
                    nonlinearity=nn.nonlinearities.sigmoid, name='dense_sigmoid_'+ptype+'_'+obj_name)
        else:
            raise

        d_final_layers[obj_name] = l_fin
        final_layers.append(l_fin)

    l_out = nn.layers.ConcatLayer(final_layers, name = 'final_concat_layer')


    metadata = utils.load_pkl(os.path.join('/home/frederic/kaggle-dsb3/dsb/storage/metadata/dsb3/models/eavsteen/',"r_elias_28-20170331-230303.pkl"))
    nn.layers.set_all_param_values(l_out, metadata['param_values'])

    features = d_final_layers['malignancy']
    print 'features layer', features.name

    return features


def build_model():
    l_in = nn.layers.InputLayer((batch_size, ) + p_transform['patch_size'])
    l_ds = nn.layers.DimshuffleLayer(l_in, pattern=[0,'x',1,2,3])

    l_target = nn.layers.InputLayer((batch_size,))

    l = conv3d(l_ds, 32)
    l = max_pool3d(l)
    l = conv3d(l, 32)
    l = conv3d(l, 64)


    l = max_pool3d(l)
    l = conv3d(l, 64)
    l = conv3d(l, 64)

    l = max_pool3d(l)
    l = conv3d(l, 64)
    l = conv3d(l, 64)

    l = max_pool3d(l)

    l = dense(drop(l), 128)

    l_out = nn.layers.DenseLayer(l, num_units=1,
        W=nn.init.Orthogonal(),
        b=nn.init.Constant(0.01),
        nonlinearity=nn.nonlinearities.sigmoid, name='final_sigmoid')


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
