import numpy as np
import data_transforms
import data_iterators
import pathfinder
from functools import partial
import utils

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (64, 64, 64),
               'mm_patch_size': (64, 64, 64),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-16, 16],
    'translation_range_y': [-16, 16],
    'translation_range_x': [-16, 16],
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}

zmuv_mean, zmuv_std = None, None


# data preparation function
def data_prep_function(data, patch_center, luna_annotations, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, **kwargs):
    x, patch_annotation_tf, annotations_tf = data_transforms.transform_patch3d(data=data,
                                                                               luna_annotations=luna_annotations,
                                                                               patch_center=patch_center,
                                                                               p_transform=p_transform,
                                                                               p_transform_augment=p_transform_augment,
                                                                               pixel_spacing=pixel_spacing,
                                                                               luna_origin=luna_origin)
    x = data_transforms.hu2normHU(x)
    # x = data_transforms.zmuv(x, zmuv_mean, zmuv_std)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment, p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

# data iterators
batch_size = 1
nbatches_chunk = 8
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.PatchPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                    batch_size=chunk_size,
                                                                    transform_params=p_transform,
                                                                    data_prep_fun=data_prep_function_train,
                                                                    rng=rng,
                                                                    patient_ids=train_pids,
                                                                    full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.ValidPatchPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                         transform_params=p_transform,
                                                                         data_prep_fun=data_prep_function_valid,
                                                                         patient_ids=valid_pids)

print 'estimating ZMUV parameters'
x_big = None
for i, (x, _, _) in zip(xrange(1), train_data_iterator.generate()):
    x_big = x if x_big is None else np.concatenate((x_big, x), axis=0)
zmuv_mean = x_big.mean()
zmuv_std = x_big.std()
print 'mean:', zmuv_mean
print 'std:', zmuv_std
