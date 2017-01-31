import numpy as np
import data_transforms
import data_iterators
import pathfinder

restart_from_save = None
rng = np.random.RandomState(42)
# p_transform = {'patch_size': (128, 128, 128),
#                'mm_patch_size': (320, 320, 320),
#                'pixel_spacing': (1., 1., 1.)
#                }
# p_transform_augment = {
#     'translation_range_z': [-10, 10],
#     'translation_range_y': [-10, 10],
#     'translation_range_x': [-10, 10],
#     'rotation_range_z': [-10, 10],
#     'rotation_range_y': [-5, 5],
#     'rotation_range_x': [-5, 5]
# }

p_transform = {'patch_size': (64, 64, 64),
               'mm_patch_size': (64, 64, 64),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-40, 40],
    'translation_range_y': [-40, 40],
    'translation_range_x': [-40, 40],
    'rotation_range_z': [-10, 10],
    'rotation_range_y': [-10, 10],
    'rotation_range_x': [-10, 10]
}

batch_size = 1
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk


# def data_prep_function_train(data, luna_annotations, pixel_spacing, luna_origin):
#     x = data_transforms.hu2normHU(data)
#     x, annotations_tf = data_transforms.transform_scan3d(data=x,
#                                                          luna_annotations=luna_annotations,
#                                                          p_transform=p_transform,
#                                                          p_transform_augment=p_transform_augment,
#                                                          pixel_spacing=pixel_spacing,
#                                                          luna_origin=luna_origin)
#     y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='cube')
#     return x, y


def data_prep_function_train(data, patch_center, luna_annotations, pixel_spacing, luna_origin, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, patch_annotation_tf, annotations_tf = data_transforms.transform_patch3d(data=x,
                                                                               luna_annotations=luna_annotations,
                                                                               patch_center=patch_center,
                                                                               p_transform=p_transform,
                                                                               p_transform_augment=p_transform_augment,
                                                                               pixel_spacing=pixel_spacing,
                                                                               luna_origin=luna_origin)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y


def data_prep_function_test(data, luna_annotations, pixel_spacing, luna_origin):
    x = data_transforms.hu2normHU(data)
    x, annotations_tf = data_transforms.transform_scan3d(data=x,
                                                         luna_annotations=luna_annotations,
                                                         p_transform=p_transform,
                                                         p_transform_augment=None,
                                                         pixel_spacing=pixel_spacing,
                                                         luna_origin=luna_origin)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='cube')
    return x, y


train_data_iterator = data_iterators.PositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                               batch_size=chunk_size,
                                                               transform_params=p_transform,
                                                               data_prep_fun=data_prep_function_train,
                                                               rng=rng,
                                                               full_batch=True, random=True, infinite=True)

test_data_iterator = data_iterators.PositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                              batch_size=chunk_size,
                                                              transform_params=p_transform,
                                                              data_prep_fun=data_prep_function_test,
                                                              rng=rng,
                                                              full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 100
learning_rate_schedule = {
    0: 0.0002,
    int(max_nchunks * 0.1): 0.0001,
    int(max_nchunks * 0.3): 0.000075,
    int(max_nchunks * 0.6): 0.00005,
    int(max_nchunks * 0.9): 0.00001
}
validate_every = 2 * nchunks_per_epoch
save_every = 2 * nchunks_per_epoch
