import os
import numpy as np
import data_transforms
import pathfinder
import utils
import utils_lung
from configuration import set_configuration, config
from utils_plots import plot_slice_3d_2, plot_2d, plot_2d_4, plot_slice_3d_3

set_configuration('test_config')


def test1():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    id2zyxd = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]
    print len(luna_data_paths)
    print id2zyxd.keys()

    for k, p in enumerate(luna_data_paths):
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        img = data_transforms.hu2normHU(img)
        id = os.path.basename(p).replace('.mhd', '')

        for nodule_zyxd in id2zyxd.itervalues():
            zyx = np.array(nodule_zyxd[:3])
            voxel_coords = utils_lung.world2voxel(zyx, origin, pixel_spacing)
            diameter_mm = nodule_zyxd[-1]
            radius_px = diameter_mm / pixel_spacing[1] / 2.
            roi_radius = (radius_px, radius_px)
            slice = img[voxel_coords[0], :, :]
            slice_prev = img[voxel_coords[0] - 1, :, :]
            slice_next = img[voxel_coords[0] + 1, :, :]
            roi_center_yx = (voxel_coords[1], voxel_coords[2])
            mask = data_transforms.make_2d_mask(slice.shape, roi_center_yx, roi_radius, masked_value=0.1)
            plot_2d(slice, mask, id, image_dir)

            plot_2d_4(slice, slice_prev, slice_next, mask, id, image_dir)

            a = [{'center': roi_center_yx, 'diameter_mm': diameter_mm}]
            p_transform = {'patch_size': (256, 256),
                           'mm_patch_size': (360, 360)}
            slice_patch, mask_patch = data_transforms.luna_transform_slice(slice, a, pixel_spacing[1:],
                                                                           p_transform, None)
            plot_2d(slice_patch, mask_patch, id, image_dir)


def test_luna3d():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    id2zyxd = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]

    # luna_data_paths = [pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.223098610241551815995595311693.mhd']
    # luna_data_paths = [pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.202811684116768680758082619196.mhd']
    # luna_data_paths = [pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.174168737938619557573021395302.mhd']
    luna_data_paths = [
        pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.mhd']
    for k, p in enumerate(luna_data_paths):
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        id = os.path.basename(p).replace('.mhd', '')
        print id

        annotations = id2zyxd[id]

        _, annotations_out = data_transforms.transform_scan3d(img,
                                                              pixel_spacing=pixel_spacing,
                                                              p_transform=config().p_transform,
                                                              p_transform_augment=None,
                                                              # config().p_transform_augment,
                                                              luna_annotations=annotations,
                                                              luna_origin=origin)

        img_out, mask = config().data_prep_function_test(img,
                                                         pixel_spacing=pixel_spacing,
                                                         luna_annotations=annotations,
                                                         luna_origin=origin,
                                                         )

        plot_slice_3d_2(img_out, mask, 0, id)
        plot_slice_3d_2(img_out, mask, 1, id)
        plot_slice_3d_2(img_out, mask, 2, id)

        mask[mask == 0.] = 0.1
        for zyxd in annotations_out:
            plot_slice_3d_2(img_out, mask, 0, id, idx=zyxd[0])
            plot_slice_3d_2(img_out, mask, 1, id, idx=zyxd[1])
            plot_slice_3d_2(img_out, mask, 2, id, idx=zyxd[2])


def count_proportion():
    id2zyxd = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]

    n_white = 0
    n_black = 0

    for k, p in enumerate(luna_data_paths):
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        img = data_transforms.hu2normHU(img)
        id = os.path.basename(p).replace('.mhd', '')
        print id

        annotations = id2zyxd[id]

        img_out, annotations_out = data_transforms.transform_scan3d(img,
                                                                    pixel_spacing=pixel_spacing,
                                                                    p_transform=config().p_transform,
                                                                    p_transform_augment=None,
                                                                    # config().p_transform_augment,
                                                                    luna_annotations=annotations,
                                                                    luna_origin=origin)

        mask = data_transforms.make_3d_mask_from_annotations(img_out.shape, annotations_out, shape='sphere')
        n_white += np.sum(mask)
        n_black += mask.shape[0] * mask.shape[1] * mask.shape[2] - np.sum(mask)

        print 'white', n_white
        print 'black', n_black


def test_kaggle3d():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_1/'
    utils.auto_make_dir(image_dir)

    patient_data_paths = utils_lung.get_patient_data_paths(pathfinder.DATA_PATH)
    print len(patient_data_paths)

    for k, p in enumerate(patient_data_paths):
        pid = utils_lung.extract_pid(p)
        sid2data, sid2metadata = utils_lung.get_patient_data(p)
        sids_sorted = utils_lung.sort_slices_plane(sid2metadata)
        sids_sorted_jonas = utils_lung.sort_slices_jonas(sid2metadata)
        sid2position = utils_lung.slice_location_finder(sid2metadata)

        try:
            slice_thickness_pos = np.abs(sid2metadata[sids_sorted[0]]['ImagePositionPatient'][2] -
                                         sid2metadata[sids_sorted[1]]['ImagePositionPatient'][2])
        except:
            print 'This patient has no ImagePosition!'
            slice_thickness_pos = 0.
        try:
            slice_thickness_loc = np.abs(
                sid2metadata[sids_sorted[0]]['SliceLocation'] - sid2metadata[sids_sorted[1]]['SliceLocation'])
        except:
            print 'This patient has no SliceLocation!'
            slice_thickness_loc = 0.

        jonas_slicethick = []
        for i in xrange(len(sids_sorted_jonas) - 1):
            s = np.abs(sid2position[sids_sorted_jonas[i + 1]] - sid2position[sids_sorted_jonas[i]])
            jonas_slicethick.append(s)

        img = np.stack([data_transforms.ct2normHU(sid2data[sid], sid2metadata[sid]) for sid in sids_sorted])
        xx = (jonas_slicethick[0],
              sid2metadata[sids_sorted[0]]['PixelSpacing'][0],
              sid2metadata[sids_sorted[0]]['PixelSpacing'][1])
        pixel_spacing = np.asarray(xx)

        img_out = data_transforms.transform_scan3d(img,
                                                   pixel_spacing=pixel_spacing,
                                                   p_transform=config().p_transform,
                                                   p_transform_augment=config().p_transform_augment)

        # plot_2d_3dimg(img_out, img_out, axis=0, pid=pid + 'z', img_dir=image_dir)
        plot_slice_3d_2(img_out, img_out, axis=1, pid=pid + 'y', img_dir=image_dir)
        # plot_2d_3dimg(img_out, img_out, axis=2, pid=pid + 'x', img_dir=image_dir)


def test_luna_patches_3d():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    id2zyxd = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]

    luna_data_paths = [
        pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.mhd']
    for k, p in enumerate(luna_data_paths):
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        # img = data_transforms.hu2normHU(img)
        id = os.path.basename(p).replace('.mhd', '')
        print id

        annotations = id2zyxd[id]

        for zyxd in annotations:
            img_out, mask = config().data_prep_function_train(img,
                                                              pixel_spacing=pixel_spacing,
                                                              p_transform=config().p_transform,
                                                              p_transform_augment=config().p_transform_augment,
                                                              patch_center=zyxd,
                                                              luna_annotations=annotations,
                                                              luna_origin=origin)
            try:
                plot_slice_3d_2(img_out, mask, 0, id)
                plot_slice_3d_2(img_out, mask, 1, id)
                plot_slice_3d_2(img_out, mask, 2, id)
            except:
                pass
        print '------------------------------------------'


if __name__ == '__main__':
    # test_kaggle3d()
    # test_luna3d()
    test_luna_patches_3d()
    # count_proportion()
