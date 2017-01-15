import matplotlib
# matplotlib.use('Qt4Agg')
import data_transforms
import numpy as np
import glob
import re
from matplotlib import animation
import matplotlib.pyplot as plt
import utils
import data_transforms as data_test
import utils_lung
import pylab
import pathfinder
import data_transforms


def test2():
    patient_path = '/mnt/sda3/data/kaggle-lung/sample_images/00cba091fa4ad62cc3200a657aeb957e'
    patient_data = utils_lung.get_patient_data(patient_path)
    patient_data = utils_lung.sort_slices(patient_data)

    full_img = np.stack([data_transforms.ct2hu(d['data'], d['metadata']) for d in patient_data])
    print full_img.shape
    print np.min(full_img)
    print np.max(full_img)

    for i in range(full_img.shape[0]):
        plt.imshow(full_img[i, :, :], cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.show()

    for i in range(36):
        plt.subplot(6, 6, i + 1)
        if 4 * i < full_img.shape[0]:
            plt.imshow(full_img[4 * i, :, :], cmap=plt.cm.bone)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()

    # for i in range(36):
    #     plt.subplot(6, 6, i + 1)
    #     if 4 * i < full_img.shape[0]:
    #         plt.imshow(full_img[:, 4 * i, :], cmap=plt.cm.bone)
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()


def test1():
    """
    :return:
    """
    patient_data_paths = utils_lung.get_patient_data_paths(pathfinder.TRAIN_DATA_PATH)
    print len(patient_data_paths)
    for p in patient_data_paths:
        patient_data = utils_lung.get_patient_data(p)
        sort_slice_instance = utils_lung.sort_slices(patient_data)
        sidx = [s['slice_id'] for s in sort_slice_instance]
        print p, len(patient_data), len(sidx)
        for slice in patient_data:
            # print slice['patient_id'], slice['slice_id'], slice['metadata']
            print np.min(slice['data']), np.max(slice['data']), np.mean(slice['data'][slice['data'] < 0]), \
                slice['metadata']['RescaleSlope'], slice['metadata']['RescaleIntercept']
        print '======================================================='


if __name__ == '__main__':
    test2()
