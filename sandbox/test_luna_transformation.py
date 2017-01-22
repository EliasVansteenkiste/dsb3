import matplotlib.pyplot as plt
import numpy as np
import pathfinder
import utils
import utils_lung
import os
import data_transforms
from configuration import set_configuration, config

set_configuration('test_config')


def plot_2d_3dimg(image3d, mask3d, axis, pid, img_dir, idx=None):
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


def test2():
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

        annotations = id2zyxd[id]

        img_out, annotations_out = data_transforms.luna_transform_scan3d(img, annotations, origin,
                                                                         pixel_spacing,
                                                                         p_transform=config().p_transform,
                                                                         p_transform_augment=config().p_transform_augment)

        mask = np.zeros_like(img_out)
        for zyxd in annotations_out:
            print zyxd
            mask += data_transforms.make_3d_mask(img_out.shape, zyxd[:3], zyxd[-1] / 2, masked_value=0.1)

        for zyxd in annotations_out:
            plot_2d_3dimg(img_out, mask, 0, id, image_dir, idx=zyxd[0])
            plot_2d_3dimg(img_out, mask, 1, id, image_dir, idx=zyxd[1])
            plot_2d_3dimg(img_out, mask, 2, id, image_dir, idx=zyxd[2])


if __name__ == '__main__':
    test2()
