import matplotlib.pyplot as plt
import numpy as np  # linear algebra

import pathfinder
import utils
import utils_lung
import os
import data_transforms


def plot_2d(image2d, pid, img_dir):
    fig = plt.figure()
    fig.canvas.set_window_title(pid)
    ax = fig.add_subplot(111)
    ax.imshow(image2d, cmap=plt.cm.gray)
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
        id = os.path.basename(p).replace('.mhd', '')
        for roi in id2zyxd[id]:
            zyx = np.array(roi[:3])
            voxel_coords = utils_lung.world2voxel(zyx, origin, spacing)
            roi_radius = 32.5
            slice_y = slice(voxel_coords[1] - roi_radius, voxel_coords[1] + roi_radius)
            slice_x = slice(voxel_coords[2] - roi_radius, voxel_coords[2] + roi_radius)
            print img.shape
            print voxel_coords[0], slice_x, slice_y
            patch = img[voxel_coords[0], slice_y, slice_x]
            patch = data_transforms.hu2normHU(patch)
            print patch.shape
            print np.min(patch), np.max(patch)
            plot_2d(patch, id, image_dir)


if __name__ == '__main__':
    test1()
