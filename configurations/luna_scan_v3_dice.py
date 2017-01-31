import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn
from collections import namedtuple
from functools import partial
import theano.tensor as T
import utils
import luna_patch1_v3_dice

restart_from_save = None
rng = np.random.RandomState(42)
# transformations
p_transform = {'patch_size': (320, 320, 320),
               'mm_patch_size': (320, 320, 320),
               'pixel_spacing': (1., 1., 1.)
               }


# data preparation function
def data_prep_function(data, luna_annotations, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, annotations_tf = data_transforms.transform_scan3d(data=x,
                                                         pixel_spacing=pixel_spacing,
                                                         p_transform=p_transform,
                                                         luna_annotations=luna_annotations,
                                                         p_transform_augment=p_transform_augment,
                                                         luna_origin=luna_origin)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y


data_prep_function_valid = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

# data iterators
batch_size = 1
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)

valid_data_iterator = data_iterators.PositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                               batch_size=1,
                                                               transform_params=p_transform,
                                                               data_prep_fun=data_prep_function_valid,
                                                               rng=rng,
                                                               patient_ids=train_valid_ids['valid'],
                                                               full_batch=False, random=False, infinite=False)


def build_model():
    patch_model = luna_patch1_v3_dice.build_model()
    # metadata
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, 'luna_patch1_v3_dice')
    metadata = utils.load_pkl(metadata_path)
    nn.layers.set_all_param_values(patch_model.l_out, metadata['param_values'])

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(patch_model.l_in, patch_model.l_out, patch_model.l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = T.flatten(nn.layers.get_output(model.l_out))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    targets = T.clip(targets, 1e-6, 1.)
    dice = (2. * T.sum(targets * predictions) + epsilon) / (T.sum(predictions) + T.sum(targets) + epsilon)
    return -1. * dice
