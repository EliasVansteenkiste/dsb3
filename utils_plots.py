import matplotlib
import utils

if utils.hostname() != 'user':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib.animation as animation
from scipy import ndimage

warnings.simplefilter('ignore')


def plot_slice_3d_2(image3d, mask, axis, pid, img_dir=None, idx=None):
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    fig.canvas.set_window_title(pid)
    mask_smoothed = np.copy(mask)
    mask_smoothed[mask == 0] = 0.1
    masked_image = image3d * mask_smoothed
    if idx is None:
        roi_idxs = np.where(mask > 0)
        if len(roi_idxs[0]) > 0:
            idx = (np.mean(roi_idxs[0]), np.mean(roi_idxs[1]), np.mean(roi_idxs[2]))
        else:
            print 'No nodules'
            idx = np.array(image3d.shape) / 2
    if axis == 0:  # sax
        ax[0, 0].imshow(image3d[idx[0], :, :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[idx[0], :, :], cmap=plt.cm.gray)
        ax[1, 0].imshow(masked_image[idx[0], :, :], cmap=plt.cm.gray)
    if axis == 1:  # 2 lungs
        ax[0, 0].imshow(image3d[:, idx[1], :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[:, idx[1], :], cmap=plt.cm.gray)
        ax[1, 0].imshow(masked_image[:, idx[1], :], cmap=plt.cm.gray)
    if axis == 2:  # side view
        ax[0, 0].imshow(image3d[:, :, idx[2]], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[:, :, idx[2]], cmap=plt.cm.gray)
        ax[1, 0].imshow(masked_image[:, :, idx[2]], cmap=plt.cm.gray)

    if img_dir is not None:
        fig.savefig(img_dir + '/%s%s.png' % (pid, axis), bbox_inches='tight')
    else:
        plt.show()
    fig.clf()
    plt.close('all')


def plot_slice_3d_3(input, mask, prediction, axis, pid, img_dir=None, idx=None):
    # to convert cuda arrays to numpy array
    input = np.asarray(input)
    mask = np.asarray(mask)
    prediction = np.asarray(prediction)

    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    fig.canvas.set_window_title(pid)
    if idx is None:
        roi_idxs = np.where(mask > 0)
        if len(roi_idxs[0]) > 0:
            idx = (np.mean(roi_idxs[0]), np.mean(roi_idxs[1]), np.mean(roi_idxs[2]))
        else:
            print 'No nodules'
            idx = np.array(input.shape) / 2
    if axis == 0:  # sax
        ax[0, 0].imshow(prediction[idx[0], :, :], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[idx[0], :, :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[idx[0], :, :], cmap=plt.cm.gray)
    if axis == 1:  # 2 lungs
        ax[0, 0].imshow(prediction[:, idx[1], :], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[:, idx[1], :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[:, idx[1], :], cmap=plt.cm.gray)
    if axis == 2:  # side view
        ax[0, 0].imshow(prediction[:, :, idx[2]], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[:, :, idx[2]], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[:, :, idx[2]], cmap=plt.cm.gray)
    if img_dir is not None:
        fig.savefig(img_dir + '/%s%s.png' % (pid, axis), bbox_inches='tight')
    else:
        plt.show()
    fig.clf()
    plt.close('all')


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


def animation(input, mask, predictions):
    pass