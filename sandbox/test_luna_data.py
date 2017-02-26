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


def test1():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    # sys.stdout = logger.Logger(image_dir + '/test_luna.log')
    # sys.stderr = sys.stdout

    id2zyxd = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]
    print len(luna_data_paths)
    print id2zyxd.keys()

    for k, p in enumerate(luna_data_paths):
        img, origin, spacing = utils_lung.read_mhd(p)
        img = data_transforms.hu2normHU(img)
        id = os.path.basename(p).replace('.mhd', '')
        for roi in id2zyxd[id]:
            zyx = np.array(roi[:3])
            voxel_coords = utils_lung.world2voxel(zyx, origin, spacing)
            print spacing
            radius_mm = roi[-1] / 2.
            radius_px = radius_mm / spacing[1]
            print 'r in pixels =', radius_px
            # roi_radius = (32.5, 32.5)
            roi_radius = (radius_px, radius_px)
            slice = img[voxel_coords[0], :, :]
            roi_center_yx = (voxel_coords[1], voxel_coords[2])
            # print slice.shape, slice_resample.shape
            mask = make_circular_mask(slice.shape, roi_center_yx, roi_radius)
            plot_2d(slice, mask, id, image_dir)

            slice_mm, _ = resample(slice, spacing[1:])
            roi_center_mm = tuple(int(r * ps) for r, ps in zip(roi_center_yx, spacing[1:]))
            mask_mm = make_circular_mask(slice_mm.shape, roi_center_mm, (radius_mm, radius_mm))
            plot_2d(slice_mm, mask_mm, id, image_dir)


def test2():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]
    print len(luna_data_paths)
    pid2mm_shape = {}

    for k, p in enumerate(luna_data_paths):
        img, origin, spacing = utils_lung.read_mhd(p)
        id = os.path.basename(p).replace('.mhd', '')
        mm_shape = img.shape * spacing
        pid2mm_shape[id] = mm_shape
        print k, id, mm_shape
        if k % 50 == 0:
            print 'Saved'
            utils.save_pkl(pid2mm_shape, image_dir + '/pid2mm.pkl')

    utils.save_pkl(pid2mm_shape, image_dir + '/pid2mm.pkl')


def test3():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    id2mm_shape = utils.load_pkl(image_dir + '/pid2mm.pkl')
    s = [(key, value) for (key, value) in sorted(id2mm_shape.items(), key=lambda x: x[1][0])]
    for i in xrange(5):
        print s[i]
    print '--------------------------'
    for i in xrange(1,6):
        print s[-i]


if __name__ == '__main__':
    # test2()
    test3()
