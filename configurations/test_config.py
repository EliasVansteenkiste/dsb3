import numpy as np

rng = np.random.RandomState(42)
patch_size = (64, 64)
mm_patch_size = (128, 128)

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
