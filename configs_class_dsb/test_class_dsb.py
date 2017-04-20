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
candidates_config = 'dsb_c3_s5_p8a1'
# TODO: import correct config here

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
n_candidates_per_patient = 6


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
batch_size = 2

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids, test_stage2_pids = train_valid_ids['training'], train_valid_ids['validation'], \
                                                      train_valid_ids['test'], train_valid_ids['test_stage2']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)
print 'n test stage 2', len(test_stage2_pids)

train_pids.extend(valid_pids)

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
                                                              patient_ids=test_pids,
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

ids_check = ['3b4c610fce3d4d723bc17986395af9ab',
             '3e4568aa1b37bd06f3917bc505ab6c2a',
             '401c2a2e7ff122ec5c558089d8ae3586',
             '419af46335739bb811e8bc97c3863836',
             '1f80571a52f38a5d9c029149612cb553',
             '3e4568aa1b37bd06f3917bc505ab6c2a',
             '3d8f006eeab0a4ea109ffe3901c7f695',
             '419af46335739bb811e8bc97c3863836',
             '417e3c40213fe0b8474b1ff74318f14c',
             '3d8f006eeab0a4ea109ffe3901c7f695']

test_stage2_data_iterator = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                                    batch_size=1,
                                                                    transform_params=p_transform,
                                                                    n_candidates_per_patient=n_candidates_per_patient,
                                                                    data_prep_fun=data_prep_function_valid,
                                                                    id2candidates_path=id2candidates_path,
                                                                    rng=rng,
                                                                    patient_ids=ids_check,
                                                                    random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / batch_size
max_nchunks = nchunks_per_epoch * 10

validate_every = int(3 * nchunks_per_epoch)
save_every = int(0.5 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 5e-6,
    int(0.5 * max_nchunks): 2e-6,
    int(0.6 * max_nchunks): 1e-6,
    int(0.7 * max_nchunks): 5e-7,
    int(0.9 * max_nchunks): 2e-7
}

untrained_weigths_grad_scale = 10
