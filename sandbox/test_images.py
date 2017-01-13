import matplotlib
# matplotlib.use('Qt4Agg')
import data
import numpy as np
import glob
import re
from matplotlib import animation
import matplotlib.pyplot as plt
import utils
import data as data_test
import utils_lung
import pylab

patient_path = '/mnt/sda3/data/kaggle-lung/sample_images/00cba091fa4ad62cc3200a657aeb957e'
patient_data = utils_lung.get_patient_data(patient_path)
patient_data = utils_lung.sort_slices(patient_data)

# for pdict in patient_data:
#     slice_img = pdict['data']
#     slice_id = pdict['slice_id']
#     print pdict['metadata']
#
#     fig = plt.figure(1)
#     fig.canvas.set_window_title(slice_id)
#     plt.subplot(121)
#     im2 = fig.gca().imshow(slice_img, pylab.cm.bone)
#     plt.show()

full_img = np.stack([d['data'] for d in patient_data])
print full_img.shape
print np.min(full_img)
print np.max(full_img)

for i in range(36):
    plt.subplot(6, 6, i + 1)
    if 4 * i < full_img.shape[0]:
        plt.imshow(full_img[4 * i, :, :], cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])
# plt.show()

for i in range(36):
    plt.subplot(6, 6, i + 1)
    if 4 * i < full_img.shape[0]:
        plt.imshow(full_img[:, 4 * i, :], cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])
plt.show()
