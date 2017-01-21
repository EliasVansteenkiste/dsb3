import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lio_transform3d import affine_transform, apply_affine_transform
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


def plot_2d_3dimg(image3d, axis, pid, img_dir):
    fig = plt.figure()
    fig.canvas.set_window_title(pid)
    ax = fig.add_subplot(111)
    idx = image3d.shape[axis] / 2
    if axis == 0:  # sax
        ax.imshow(image3d[idx, :, :], cmap=plt.cm.gray)
    if axis == 1:  # 2 lungs
        ax.imshow(image3d[:, idx, :], cmap=plt.cm.gray)
    if axis == 2:  # side view
        ax.imshow(image3d[:, :, idx], cmap=plt.cm.gray)
    plt.show()
    fig.savefig(img_dir + '/%s.png' % pid, bbox_inches='tight')
    fig.clf()
    plt.close('all')


def draw_circles(img):
    rr, cc = skimage.draw.circle(0, 0, 10)
    img[rr, cc, :] = 1.

    rr, cc = skimage.draw.circle(img.shape[0] - 1, img.shape[1] - 1, 10)
    for r, c in zip(rr, cc):
        try:
            img[r, c, :] = 1.
        except:
            pass

    rr, cc = skimage.draw.circle(0, img.shape[1] - 1, 10)
    for r, c in zip(rr, cc):
        try:
            img[r, c, :] = 1.
        except:
            pass

    rr, cc = skimage.draw.circle(img.shape[0] - 1, 0, 10)
    for r, c in zip(rr, cc):
        try:
            img[r, c, :] = 1.
        except:
            pass
    return img


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
        img = np.swapaxes(img, 0, -1)
        pixel_spacing = np.array([pixel_spacing[-1], pixel_spacing[1], pixel_spacing[0]])
        print img.shape
        print pixel_spacing
        img = draw_circles(img)

        input_shape = np.asarray(img.shape)
        mm_shape = input_shape * pixel_spacing
        output_shape = np.asarray((128, 128, 128))
        print mm_shape

        mm_patch_size = np.asarray((256, 256, 256), np.float)
        mm_patch_origin = np.array((mm_shape[0] / 2 - mm_patch_size[0] / 2,
                                    mm_shape[1] / 2 - mm_patch_size[1] / 2,
                                    mm_shape[2] / 2 - mm_patch_size[2] / 2))

        mm_scale_tf = affine_transform(scale=mm_shape / input_shape)
        shift_center = affine_transform(translation=(-mm_shape[0] / 2, -mm_shape[1] / 2, -mm_shape[2] / 2))
        augment = affine_transform(rotation=(0, 0, 45))
        shift_uncenter = affine_transform(translation=(mm_shape[0] / 2, mm_shape[1] / 2, mm_shape[2] / 2))
        shift_uncenter2 = affine_transform(translation=-mm_patch_origin)
        patch_scale_tf = affine_transform(scale=output_shape / mm_patch_size)

        matrix = mm_scale_tf.dot(shift_center).dot(augment).dot(shift_uncenter).dot(shift_uncenter2).dot(patch_scale_tf)
        # matrix = normscale.dot(shift_center).dot(augment).dot(shift_uncenter)

        img_rescale = apply_affine_transform(img, matrix, order=1, output_shape=output_shape)

        plot_2d_3dimg(img_rescale, 2, id, image_dir)
        plot_2d_3dimg(img_rescale, 1, id, image_dir)
        plot_2d_3dimg(img_rescale, 0, id, image_dir)


if __name__ == '__main__':
    test1()
