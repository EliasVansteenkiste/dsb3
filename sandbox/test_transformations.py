import matplotlib.pyplot as plt
import numpy as np
import pathfinder
import utils
import utils_lung
import os
import data_transforms
from configuration import set_configuration, config

set_configuration('test_config')


def plot_2d_3dimg(image3d, mask3d, axis, pid, img_dir=None, idx=None):
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    fig.canvas.set_window_title(pid)
    idx = image3d.shape[axis] / 2 if idx is None else idx
    if axis == 0:  # sax
        ax[0, 0].imshow(image3d[idx, :, :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask3d[idx, :, :], cmap=plt.cm.gray)
        ax[1, 0].imshow(image3d[idx, :, :] * mask3d[idx, :, :], cmap=plt.cm.gray)
    if axis == 1:  # 2 lungs
        ax[0, 0].imshow(image3d[:, idx, :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask3d[:, idx, :], cmap=plt.cm.gray)
        ax[1, 0].imshow(image3d[:, idx, :] * mask3d[:, idx, :], cmap=plt.cm.gray)
    if axis == 2:  # side view
        ax[0, 0].imshow(image3d[:, :, idx], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask3d[:, :, idx], cmap=plt.cm.gray)
        ax[1, 0].imshow(image3d[:, :, idx] * mask3d[:, :, idx], cmap=plt.cm.gray)
    plt.show()
    if img_dir is not None:
        fig.savefig(img_dir + '/%s.png' % pid, bbox_inches='tight')
    fig.clf()
    plt.close('all')


def plot_2d(img, mask, pid, img_dir):
    # fig = plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    fig.canvas.set_window_title(pid)
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 1].imshow(mask, cmap='gray')
    ax[1, 0].imshow(img * mask, cmap='gray')
    plt.show()
    fig.savefig(img_dir + '/%s.png' % pid, bbox_inches='tight')
    fig.clf()
    plt.close('all')


def plot_2d_4(img, img_prev, img_next, mask, pid, img_dir):
    # fig = plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    fig.canvas.set_window_title(pid)
    ax[0, 0].imshow(img, cmap='gray')
    ax[0, 1].imshow(img_prev, cmap='gray')
    ax[1, 0].imshow(img_next, cmap='gray')
    ax[1, 1].imshow(img * mask, cmap='gray')
    plt.show()
    fig.savefig(img_dir + '/%s.png' % pid, bbox_inches='tight')
    fig.clf()
    plt.close('all')


def test1():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.automakedir(image_dir)

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
            mask = data_transforms.make_roi_mask(slice.shape, roi_center_yx, roi_radius, masked_value=0.1)
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
    utils.automakedir(image_dir)

    id2zyxd = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]

    # luna_data_paths = [pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.249530219848512542668813996730.mhd']
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

        plot_2d_3dimg(img_out, mask, 0, id)
        plot_2d_3dimg(img_out, mask, 1, id)
        plot_2d_3dimg(img_out, mask, 2, id)

        for zyxd in annotations_out:
            plot_2d_3dimg(img_out, mask, 0, id, idx=zyxd[0])
            plot_2d_3dimg(img_out, mask, 1, id, idx=zyxd[1])
            plot_2d_3dimg(img_out, mask, 2, id, idx=zyxd[2])


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
    utils.automakedir(image_dir)

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
        plot_2d_3dimg(img_out, img_out, axis=1, pid=pid + 'y', img_dir=image_dir)
        # plot_2d_3dimg(img_out, img_out, axis=2, pid=pid + 'x', img_dir=image_dir)


if __name__ == '__main__':
    # test_kaggle3d()
    test_luna3d()
    # count_proportion()
