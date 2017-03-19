import data_transforms
import data_iterators
import pathfinder
import utils
import string
import numpy as np
import lasagne as nn
import lung_segmentation
import utils_lung

# calculate the following things correctly!
p_transform = {'patch_size': (416, 416, 416),
               'mm_patch_size': (416, 416, 416),
               'pixel_spacing': (1, 1, 1)
               }


def data_prep_function(data, lung_mask, luna_annotations, pixel_spacing, luna_origin,
                       p_transform=p_transform,
                       p_transform_augment=None):
    print lung_mask.shape
    annotatations_out = []
    for zyxd in luna_annotations:
        zyx = np.array(zyxd[:3])
        voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing)
        zyxd_out = np.rint(np.append(voxel_coords, zyxd[-1]))
        annotatations_out.append(zyxd_out)
    annotatations_out = np.asarray(annotatations_out)

    lung_mask = np.asarray(lung_mask > 0, dtype='int8')
    return lung_mask, lung_mask, lung_mask, annotatations_out, None


train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

data_iterator = data_iterators.LunaScanPositiveLungMaskDataGenerator2(data_path=pathfinder.LUNA_DATA_PATH,
                                                                      lung_masks_path=pathfinder.LUNA_LUNG_SEG_PATH,
                                                                      transform_params=p_transform,
                                                                      data_prep_fun=data_prep_function,
                                                                      rng=np.random.RandomState(42),
                                                                      batch_size=1,
                                                                      full_batch=True,
                                                                      patient_ids=valid_pids,
                                                                      random=False, infinite=False)
