import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import scipy
import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import data_transforms
import pathfinder
import utils_lung


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


def plot_2d(image3d, axis, name):
    fig = plt.figure(0)
    fig.canvas.set_window_title(name)
    idx = image3d.shape[axis] / 2
    if axis == 0:  # sax
        plt.imshow(image3d[idx, :, :], cmap=plt.cm.gray)
    if axis == 1:  # 2 lungs
        plt.imshow(image3d[:, idx, :], cmap=plt.cm.gray)
    if axis == 2:  # side view
        plt.imshow(image3d[:, :, idx], cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    plt.show()


def test1():
    patient_data_paths = utils_lung.get_patient_data_paths(pathfinder.TRAIN_DATA_PATH)
    print len(patient_data_paths)

    for p in patient_data_paths:
        sid2data, sid2metadata = utils_lung.get_patient_data(p)
        sids_sorted = utils_lung.sort_slices_plane(sid2metadata)

        slice_thickness = np.abs(
            sid2metadata[sids_sorted[0]]['SliceLocation'] - sid2metadata[sids_sorted[1]]['SliceLocation'])

        full_img = np.stack([data_transforms.ct2hu(sid2data[sid], sid2metadata[sid]) for sid in sids_sorted])
        spacing = sid2metadata[sids_sorted[0]]['PixelSpacing']
        spacing = [slice_thickness, spacing[0], spacing[1]]
        # resampled_image, _ = resample(full_img, spacing)
        print p
        plot_2d(full_img, axis=1, name=utils_lung.extract_pid(p))


if __name__ == '__main__':
    test1()
