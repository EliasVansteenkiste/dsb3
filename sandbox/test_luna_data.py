import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import scipy
import scipy.ndimage
import data_transforms
import pathfinder
import utils
import utils_lung
import logger
import sys


def plot_2d(image3d, axis, pid, img_dir):
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
    fig.savefig(img_dir + '/%s.png' % pid, bbox_inches='tight')
    fig.clf()
    plt.close('all')


def test1():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.automakedir(image_dir)

    # sys.stdout = logger.Logger(image_dir + '/test_luna.log')
    # sys.stderr = sys.stdout

    id2xyzd = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]
    print len(luna_data_paths)

    for k, p in enumerate(luna_data_paths):
        img, origin, spacing = utils_lung.read_mhd(p)
        id = ''
        xyz = np.array(id2xyzd[id][:3])
        voxel_coords = utils_lung.world2voxel(xyz, origin, spacing)
        print img.shape


if __name__ == '__main__':
    test1()
