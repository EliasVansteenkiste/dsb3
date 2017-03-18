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


def data_prep_function(data, luna_annotations, pixel_spacing, luna_origin,
                       p_transform=p_transform,
                       p_transform_augment=None):
    # make sure the data is processed the same way
    lung_mask = lung_segmentation.segment_HU_scan_ira(data)

    annotatations_out = []
    for zyxd in luna_annotations:
        zyx = np.array(zyxd[:3])
        voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing)
        zyxd_out = np.rint(np.append(voxel_coords, zyxd[-1]))
        annotatations_out.append(zyxd_out)
    annotatations_out = np.asarray(annotatations_out)

    return lung_mask, lung_mask, lung_mask, annotatations_out, None


data_iterator = data_iterators.LunaScanPositiveLungMaskDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                     transform_params=p_transform,
                                                                     data_prep_fun=data_prep_function,
                                                                     rng=np.random.RandomState(42),
                                                                     batch_size=1,
                                                                     full_batch=True,
                                                                     random=False, infinite=False)
