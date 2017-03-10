import data_transforms
import data_iterators
import pathfinder
import configs_seg_patch.luna_p_local as patch_config
import lung_segmentation

# rng = patch_config.rng
# p_transform_patch = patch_config.p_transform
#
# p_transform = {'patch_size': (416, 416, 416),
#                'mm_patch_size': (416, 416, 416),
#                'pixel_spacing': patch_config.p_transform['pixel_spacing']
#                }
#
#
# def data_prep_function(data, pixel_spacing, p_transform=p_transform):
#     # MAKE SURE THAT DATA IS PREPROCESSED THE SAME WAY
#     x, tf_matrix = data_transforms.transform_scan3d(data=data,
#                                                     pixel_spacing=pixel_spacing,
#                                                     p_transform=p_transform,
#                                                     p_transform_augment=None)
#     x = data_transforms.pixelnormHU(x)
#     return x, tf_matrix
#
#
# data_iterator = data_iterators.DSBScanDataGenerator(data_path=pathfinder.DATA_PATH,
#                                                     transform_params=p_transform,
#                                                     data_prep_fun=data_prep_function)



rng = patch_config.rng

# calculate the following things correctly!
p_transform = {'patch_size': (416, 416, 416),
               'mm_patch_size': (416, 416, 416),
               'pixel_spacing': patch_config.p_transform['pixel_spacing']
               }
window_size = 160
stride = 128
n_windows = (p_transform['patch_size'][0] - window_size) / stride + 1

valid_pids = patch_config.valid_pids


def data_prep_function(data, luna_annotations, pixel_spacing, luna_origin,
                       p_transform=p_transform,
                       p_transform_augment=None):
    # make sure the data is processed the same way
    lung_mask = lung_segmentation.segment_HU_scan(data)
    x, annotations_tf, tf_matrix, lung_mask_out = data_transforms.transform_scan3d(data=data,
                                                                                   pixel_spacing=pixel_spacing,
                                                                                   p_transform=p_transform,
                                                                                   luna_annotations=luna_annotations,
                                                                                   p_transform_augment=None,
                                                                                   luna_origin=luna_origin,
                                                                                   lung_mask=lung_mask)
    x = data_transforms.pixelnormHU(x)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y, lung_mask_out, annotations_tf, tf_matrix


valid_data_iterator = data_iterators.LunaScanPositiveLungMaskDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           transform_params=p_transform,
                                                                           data_prep_fun=data_prep_function,
                                                                           rng=rng,
                                                                           batch_size=1,
                                                                           patient_ids=valid_pids,
                                                                           full_batch=True,
                                                                           random=False, infinite=False)
