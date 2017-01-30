import matplotlib.pyplot as plt
import numpy as np
import pathfinder
import utils
import utils_lung
import os
import data_transforms
from configuration import set_configuration, config

set_configuration('test_config2')


def plot_2d_3dimg(image3d, mask3d, axis, pid, img_dir=None, idx=None):
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
    if img_dir is not None:
        fig.savefig(img_dir + '/%s.png' % pid, bbox_inches='tight')
    fig.clf()
    plt.close('all')


data_iter = config().test_data_iterator
for (x_batch, y_batch, pid_batch) in data_iter.generate():

    print pid_batch
    id = pid_batch[0]
    for i in xrange(x_batch.shape[0]):
        print x_batch.shape, y_batch.shape
        plot_2d_3dimg(x_batch[i, 0], y_batch[i, 0], 0, id)
        plot_2d_3dimg(x_batch[i, 0], y_batch[i, 0], 1, id)
        plot_2d_3dimg(x_batch[i, 0], y_batch[i, 0], 2, id)


# data_iter = config().test_data_iterator
# black, white = 0, 0
# print data_iter.nsamples
# for (x_batch, y_batch, pid_batch) in data_iter.generate():
#     id = pid_batch[0]
#     for i in xrange(x_batch.shape[0]):
#         white += np.sum(y_batch)
#         black += y_batch.size - np.sum(y_batch)
#     print white, black, black / white
