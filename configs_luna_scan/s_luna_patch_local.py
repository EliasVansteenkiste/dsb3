import data_transforms
import data_iterators
import pathfinder
import utils
import string
import numpy as np
import lasagne as nn
import configs_luna_patch.luna_patch_local as patch_config

rng = patch_config.rng
p_transform_patch = patch_config.p_transform

p_transform = {'patch_size': (448, 448, 448),
               'mm_patch_size': (448, 448, 448),
               'pixel_spacing': p_transform_patch['pixel_spacing']
               }

valid_pids = patch_config.valid_pids


def data_prep_function(data, luna_annotations, pixel_spacing, luna_origin,
                       p_transform=p_transform,
                       p_transform_augment=None):
    x = data_transforms.hu2normHU(data)
    x, annotations_tf = data_transforms.transform_scan3d(data=x,
                                                         pixel_spacing=pixel_spacing,
                                                         p_transform=p_transform,
                                                         luna_annotations=luna_annotations,
                                                         p_transform_augment=None,
                                                         luna_origin=luna_origin)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y, annotations_tf


valid_data_iterator = data_iterators.PositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                               transform_params=p_transform,
                                                               data_prep_fun=data_prep_function,
                                                               rng=rng,
                                                               batch_size=1,
                                                               patient_ids=valid_pids,
                                                               full_batch=True,
                                                               random=False, infinite=False)
