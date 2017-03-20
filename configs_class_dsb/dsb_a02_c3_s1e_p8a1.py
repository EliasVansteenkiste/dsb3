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
conv3 = partial(dnn.Conv3DDNNLayer,
                pad="valid",
                filter_size=3,
                nonlinearity=nn.nonlinearities.rectify,
                b=nn.init.Constant(0.1),
                W=nn.init.Orthogonal("relu"))

max_pool = partial(dnn.MaxPool3DDNNLayer,
                   pool_size=2)


def dense_prelu_layer(l_in, num_units):
    l = nn.layers.DenseLayer(l_in, num_units=num_units, W=nn.init.Orthogonal(),
                             nonlinearity=nn.nonlinearities.linear)
    l = nn.layers.ParametricRectifierLayer(l)
    return l


def build_nodule_classification_model(l_in):
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, patch_class_config.__name__.split('.')[-1])
    metadata = utils.load_pkl(metadata_path)

    model = patch_class_config.build_model(l_in)
    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])
    # nn_lung.remove_trainable_parameters(model.l_out)
    return model


def build_model():
    l_in = nn.layers.InputLayer((batch_size, n_candidates_per_patient, 1,) + p_transform['patch_size'])
    l_in_rshp = nn.layers.ReshapeLayer(l_in, (-1, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((batch_size,))

    nodule_classification_model = build_nodule_classification_model(l_in_rshp)

    l_roi_out = nn.layers.SliceLayer(nodule_classification_model.l_out, indices=1, axis=-1)
    l_roi_out = nn.layers.ReshapeLayer(l_roi_out, (-1, n_candidates_per_patient))

    l_out = nn.layers.FeaturePoolLayer(l_roi_out, pool_size=n_candidates_per_patient, pool_function=T.max)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target', 'l_roi_out'])(l_in, l_out, l_target, l_roi_out)


def build_objective(model, deterministic=False, epsilon=1e-12):
    targets = nn.layers.get_output(model.l_target)

    # for negative examples
    p0 = nn.layers.get_output(model.l_roi_out, deterministic=deterministic)
    p0 = T.clip(p0, epsilon, 1. - epsilon)
    p0 = T.mean(T.log(1. - p0), axis=-1)

    # for positive examples
    predictions_1 = nn.layers.get_output(model.l_out, deterministic=deterministic)[:, 0]
    p1 = T.clip(predictions_1, epsilon, 1. - epsilon)
    p1 = T.log(p1)

    loss = -1. * T.mean((1 - targets) * p0 + targets * p1, axis=0)
    return loss


def build_objective2(model, deterministic=False, epsilon=1e-12):
    targets = T.flatten(nn.layers.get_output(model.l_target))

    predictions_roi = nn.layers.get_output(model.l_roi_out, deterministic=deterministic)
    predictions_roi = T.clip(predictions_roi, epsilon, 1. - epsilon)
    predictions_roi_0 = T.mean(T.log(predictions_roi), axis=-1)

    predictions_out = T.flatten(nn.layers.get_output(model.l_out, deterministic=deterministic))
    predictions_out = T.clip(predictions_out, epsilon, 1. - epsilon)

    loss = (1 - targets) * predictions_roi_0 - targets * T.log(predictions_out)
    return T.mean(loss)


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
