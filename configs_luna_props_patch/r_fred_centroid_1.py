import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn

from collections import OrderedDict, namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import lasagne
import theano.tensor as T
import utils

restart_from_save = False
rng = np.random.RandomState(33)

# transformations
p_transform = {'patch_size': (48, 48, 48),
               'mm_patch_size': (48, 48, 48),
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

positive_proportion = 0.5

properties = ['diameter', 'calcification', 'lobulation', 'malignancy', 'margin', 'sphericity',
              'spiculation', 'texture']
nproperties = len(properties)


def label_prep_function(annotation,properties_included):
    patch_zyxd = annotation[:4]
    if patch_zyxd[-1] == 0:
        if len(properties_included)>0:
            return np.asarray([0] * len(properties_included), dtype='float32')
        else:
            return np.asarray([0] * len(properties), dtype='float32')
    else:
        label = []
        properties_dict = annotation[-1]
        if len(properties_included)>0:
            for p in properties_included:
                label.append(properties_dict[p]/5.0)
        else:
            for p in properties:
                    label.append(properties_dict[p])
    return label



# data preparation function
def data_prep_function(patch_center, pixel_spacing, luna_origin,  centroid, world_coord_system,**kwargs):
    patch_centroid_diff = data_transforms.transform_centroiddiff(

                                                               patch_center=patch_center,
                                                               pixel_spacing=pixel_spacing,
                                                               luna_origin=luna_origin,
                                                               world_coord_system=world_coord_system,
                                                               centroid=centroid)
    return patch_centroid_diff


data_prep_function_train = partial(data_prep_function, world_coord_system=True)
data_prep_function_valid = partial(data_prep_function, world_coord_system=True)

# data iterators
batch_size = 16
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']


train_data_iterator = data_iterators.CandidatesPropertiesLunaXYZDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           batch_size=chunk_size,
                                                                           transform_params=p_transform,
                                                                           label_prep_fun=label_prep_function,
                                                                           nproperties=nproperties,
                                                                           data_prep_fun=data_prep_function_train,
                                                                           rng=rng,
                                                                           patient_ids=train_pids,
                                                                           full_batch=True, random=True, infinite=True,
                                                                           positive_proportion=positive_proportion,
                                                                           random_negative_samples=True,
                                                                           properties_included=["malignancy"])

valid_data_iterator = data_iterators.CandidatesLunaValidXYZDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                      transform_params=p_transform,
                                                                      data_prep_fun=data_prep_function_valid,
                                                                      patient_ids=valid_pids,
                                                                      label_prep_fun=label_prep_function,
                                                                      properties_included=["malignancy"])

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 100

validate_every = int(5 * nchunks_per_epoch)
save_every = int(1. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-3,
    int(max_nchunks * 0.5): 1e-4,
    int(max_nchunks * 0.75): 1e-5

}




def build_model():
    l_in = nn.layers.InputLayer((None,3))
    l_target = nn.layers.InputLayer((None, 1))

    l = nn.layers.DenseLayer(l_in,50,W=nn.init.Orthogonal("relu"),nonlinearity=nn.nonlinearities.very_leaky_rectify)
    l = nn.layers.DenseLayer(l, 50, W=nn.init.Orthogonal("relu"), nonlinearity=nn.nonlinearities.very_leaky_rectify)

    l_out = nn.layers.DenseLayer(l,1,nonlinearity=nn.nonlinearities.sigmoid)


    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)




def build_objective(model, deterministic=False):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    loss = lasagne.objectives.binary_crossentropy(predictions,targets)
    return T.mean(loss)


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
