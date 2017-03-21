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
import nn_lung

# TODO: import correct config here
candidates_config = 'dsb_c3_s1e_p8a1'
# TODO: import correct config here
import configs_fpred_patch.luna_c3 as patch_class_config

restart_from_save = None
rng = np.random.RandomState(42)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
candidates_path = predictions_dir + '/%s' % candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

print 'READING CANDIDATES FROM:', candidates_path
# transformations
p_transform = {'patch_size': (48, 48, 48),
               'mm_patch_size': (48, 48, 48),
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
batch_size = 4

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
    int(5 * nchunks_per_epoch): 2e-6,
    int(6 * nchunks_per_epoch): 1e-6,
    int(7 * nchunks_per_epoch): 5e-7,
    int(9 * nchunks_per_epoch): 2e-7
}


# model

def build_nodule_classification_model(l_in):
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, patch_class_config.__name__.split('.')[-1])
    metadata = utils.load_pkl(metadata_path)

    model = patch_class_config.build_model(l_in)
    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])

    l_out = nn.layers.DenseLayer(model.l_out.input_layer,
                                 num_units=1,
                                 W=nn.init.Constant(0.),
                                 b=nn.init.Constant(np.log(0.74 / 0.26)),
                                 nonlinearity=nn.nonlinearities.sigmoid)
    model = namedtuple('Model', ['l_in', 'l_out', 'l_target'])(model.l_in, l_out, model.l_target)
    return model


def build_model():
    l_in = nn.layers.InputLayer((batch_size, n_candidates_per_patient, 1,) + p_transform['patch_size'])
    l_in_rshp = nn.layers.ReshapeLayer(l_in, (-1, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((batch_size,))

    nodule_classification_model = build_nodule_classification_model(l_in_rshp)

    l_roi_p0 = nn.layers.ReshapeLayer(nodule_classification_model.l_out,
                                      (-1, n_candidates_per_patient))

    l_out = nn_lung.ComplementProbAggregationLayer(l_roi_p0)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target', 'l_roi_p0'])(l_in, l_out, l_target, l_roi_p0)


def build_objective(model, deterministic=False, epsilon=1e-12):
    if deterministic:
        loss = build_validation_objective(model, deterministic=True, epsilon=1e-12)
        return loss
    else:
        targets = nn.layers.get_output(model.l_target)

        # for negative examples
        p0 = nn.layers.get_output(model.l_roi_p0, deterministic=deterministic)
        p0 = T.clip(p0, epsilon, 1.)
        p0 = T.sum(T.log(p0), axis=-1)

        # for positive examples
        p1 = nn.layers.get_output(model.l_out, deterministic=deterministic)[:, 0]
        p1 = T.log(p1)

        loss = -1. * T.mean((1 - targets) * p0 + targets * p1, axis=0)
    return loss


def build_validation_objective(model, deterministic=False, epsilon=1e-12):
    targets = nn.layers.get_output(model.l_target)
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)[:, 0]
    predictions = T.clip(predictions, epsilon, 1. - epsilon)
    p1 = T.log(predictions)
    p0 = T.log(1. - predictions)

    loss = -1. * T.mean((1 - targets) * p0 + targets * p1, axis=0)
    return loss


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
