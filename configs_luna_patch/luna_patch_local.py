import numpy as np
import data_transforms
import data_iterators
import pathfinder
import utils
from functools import partial

restart_from_save = None
rng = np.random.RandomState(42)
# transformations
p_transform = {'patch_size': (64, 64, 64),
               'mm_patch_size': (64, 64, 64),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-27, 27],
    'translation_range_y': [-27, 27],
    'translation_range_x': [-27, 27],
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}


# data preparation function
def data_prep_function(data, patch_center, luna_annotations, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, patch_annotation_tf, annotations_tf = data_transforms.transform_patch3d(data=x,
                                                                               luna_annotations=luna_annotations,
                                                                               patch_center=patch_center,
                                                                               p_transform=p_transform,
                                                                               p_transform_augment=p_transform_augment,
                                                                               pixel_spacing=pixel_spacing,
                                                                               luna_origin=luna_origin)
    gaussian_nodule_annotation = data_transforms.make_gaussian_annotation(patch_annotation_tf,
                                                                          patch_size=p_transform['patch_size'])
    return x, gaussian_nodule_annotation


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform)

# data iterators
batch_size = 1
nbatches_chunk = 4
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.PatchCentersPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           batch_size=chunk_size,
                                                                           transform_params=p_transform,
                                                                           data_prep_fun=data_prep_function_train,
                                                                           rng=rng,
                                                                           patient_ids=train_valid_ids['train'],
                                                                           full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.PatchCentersPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           batch_size=1,
                                                                           transform_params=p_transform,
                                                                           data_prep_fun=data_prep_function_valid,
                                                                           rng=rng,
                                                                           patient_ids=train_valid_ids['valid'],
                                                                           full_batch=False, random=False,
                                                                           infinite=False)
