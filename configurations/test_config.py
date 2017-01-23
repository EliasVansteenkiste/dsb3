import numpy as np
import data_transforms
import data_iterators
import pathfinder

restart_from_save = None
rng = np.random.RandomState(42)
p_transform = {'patch_size': (128, 128, 128),
               'mm_patch_size': (360, 400, 400),
               'pixel_spacing': (1., 1., 1.)
               }

p_transform_augment = {
    'translation_range_z': [-10, 10],
    'translation_range_y': [-10, 10],
    'translation_range_x': [-10, 10],
    'rotation_range_z': [-10, 10],
    'rotation_range_y': [-5, 5],
    'rotation_range_x': [-5, 5]
}

batch_size = 1
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk


def data_prep_function_train(data, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, annotations = data_transforms.transform_scan3d(data=x, p_transform=p_transform,
                                                      p_transform_augment=p_transform_augment, **kwargs)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations, shape='sphere')
    return x, y


def data_prep_function_test(data, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, annotations = data_transforms.transform_scan3d(data=x, p_transform=p_transform,
                                                      p_transform_augment=None, **kwargs)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations, shape='sphere')
    return x, y


train_data_iterator = data_iterators.LunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                       batch_size=chunk_size,
                                                       transform_params=p_transform,
                                                       data_prep_fun=data_prep_function_train,
                                                       rng=rng,
                                                       full_batch=True, random=True, infinite=True)

test_data_iterator = data_iterators.LunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
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
