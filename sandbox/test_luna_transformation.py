import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pathfinder
import utils
import utils_lung
import os
import data_transforms
import skimage.draw
import scipy
import collections
from collections import defaultdict


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

        # find nodules with equal z coordinate
        z2nodule_zyxd = defaultdict(list)
        for i, nodule_zyxd in enumerate(id2zyxd[id]):
            z = nodule_zyxd[0]
            z2nodule_zyxd[z].append(nodule_zyxd)

        for nodules_zyxd in z2nodule_zyxd.itervalues():
            if len(nodules_zyxd) == 1:
                nodule_zyxd = nodules_zyxd[0]

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
            slice_patch, mask_patch = data_transforms.luna_transform_rescale_slice(slice, a, pixel_spacing[1:],
                                                                                   p_transform, None)
            plot_2d(slice_patch, mask_patch, id, image_dir)


if __name__ == '__main__':
    test1()
