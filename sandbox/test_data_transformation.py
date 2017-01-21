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


def make_circular_mask(img_shape, roi_center, roi_radii):
    mask = np.ones(img_shape) * 0.1
    rr, cc = skimage.draw.ellipse(roi_center[0], roi_center[1], roi_radii[0], roi_radii[1], img_shape)
    mask[rr, cc] = 1.
    return mask


def resample(image, spacing, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array(spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing


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


def test1():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.automakedir(image_dir)

    # sys.stdout = logger.Logger(image_dir + '/test_luna.log')
    # sys.stderr = sys.stdout

    id2zyxd = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]
    print len(luna_data_paths)
    print id2zyxd.keys()

    for k, p in enumerate(luna_data_paths):
        img, origin, spacing = utils_lung.read_mhd(p)
        img = data_transforms.hu2normHU(img)
        id = os.path.basename(p).replace('.mhd', '')

        # data_mm, _ = resample(img, spacing)
        # print data_mm.shape
        # plot_2d_3dimg(data_mm, 0, id, image_dir)
        # plot_2d_3dimg(data_mm, 1, id, image_dir)
        # plot_2d_3dimg(data_mm, 2, id, image_dir)

        data_mm = data_transforms.luna_transform_rescale_scan(img, spacing, p_transform={'patch_size': (256, 256, 256)})
        print data_mm.shape
        plot_2d_3dimg(data_mm, 0, id, image_dir)
        plot_2d_3dimg(data_mm, 1, id, image_dir)
        plot_2d_3dimg(data_mm, 2, id, image_dir)


if __name__ == '__main__':
    test1()
