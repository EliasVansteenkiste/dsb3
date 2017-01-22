import numpy as np
import data_transforms

rng = np.random.RandomState(42)
p_transform = {'patch_size': (256, 256, 256),
               'mm_patch_size': (360, 360, 360),
               'pixel_spacing': (1., 1., 1.)
               }

p_transform_augment = {
    'translation_range_z': [-10, 10],
    'translation_range_y': [-10, 10],
    'translation_range_x': [-10, 10],
    'rotation_range_z': [-45, 45],
    'rotation_range_y': [0, 0],
    'rotation_range_x': [0, 0]
}


def data_prep_function(data, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, annotations = data_transforms.transform_scan3d(data=x, p_transform=p_transform,
                                                      p_transform_augment=p_transform_augment, **kwargs)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations, shape='sphere')
    return x, y
