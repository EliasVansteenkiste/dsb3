import data_transforms
import data_iterators
import pathfinder
import configs_seg_patch.luna_p_local as patch_config

rng = patch_config.rng
p_transform_patch = patch_config.p_transform

p_transform = {'patch_size': (416, 416, 416),
               'mm_patch_size': (416, 416, 416),
               'pixel_spacing': patch_config.p_transform['pixel_spacing']
               }


def data_prep_function(data, pixel_spacing, p_transform=p_transform):
    # MAKE SURE THAT DATA IS PREPROCESSED THE SAME WAY
    x, tf_matrix = data_transforms.transform_scan3d(data=data,
                                                    pixel_spacing=pixel_spacing,
                                                    p_transform=p_transform,
                                                    p_transform_augment=None)
    x = data_transforms.pixelnormHU(x)
    return x, tf_matrix


data_iterator = data_iterators.DSBScanDataGenerator(data_path=pathfinder.DATA_PATH,
                                                    transform_params=p_transform,
                                                    data_prep_fun=data_prep_function)
