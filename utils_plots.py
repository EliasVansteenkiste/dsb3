import matplotlib
import utils

if utils.hostname() != 'user':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib.animation as animation

warnings.simplefilter('ignore')
anim_running = True


def plot_slice_3d_2(image3d, mask, axis, pid, img_dir=None, idx=None):
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    fig.canvas.set_window_title(pid)
    masked_image = image3d * mask
    if idx is None:
        roi_idxs = np.where(mask == 1.)
        if len(roi_idxs[0]) > 0:
            idx = (np.mean(roi_idxs[0]), np.mean(roi_idxs[1]), np.mean(roi_idxs[2]))
        else:
            print 'No nodules'
            idx = np.array(image3d.shape) / 2
    else:
        idx = idx.astype(int)
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
            idx = (int(np.mean(roi_idxs[0])),
                   int(np.mean(roi_idxs[1])),
                   int(np.mean(roi_idxs[2])))
        else:
            print 'No nodules'
            idx = np.array(input.shape) / 2
    else:
        idx = idx.astype(int)
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
        fig.savefig(img_dir + '/%s-%s.png' % (pid, axis), bbox_inches='tight')
    else:
        plt.show()
    fig.clf()
    plt.close('all')


def plot_slice_3d_3axis(input, pid, img_dir=None, idx=None):
    # to convert cuda arrays to numpy array
    input = np.asarray(input)
    if idx == None:
        idx = np.array(input.shape)/2

    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    fig.canvas.set_window_title(pid)
    ax[0, 0].imshow(input[idx[0], :, :], cmap=plt.cm.gray)
    ax[1, 0].imshow(input[:, idx[1], :], cmap=plt.cm.gray)
    ax[0, 1].imshow(input[:, :, idx[2]], cmap=plt.cm.gray)

    if img_dir is not None:
        fig.savefig(img_dir + '/%s.png' % (pid), bbox_inches='tight')
    else:
        plt.show()
    fig.clf()
    plt.close('all')


def plot_slice_3d_4(input, mask, prediction, lung_mask, axis, pid, img_dir=None, idx=None):
    # to convert cuda arrays to numpy array
    input = np.asarray(input)
    mask = np.asarray(mask)
    prediction = np.asarray(prediction)

    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    fig.canvas.set_window_title(pid)
    if idx is None:
        roi_idxs = np.where(mask > 0)
        if len(roi_idxs[0]) > 0:
            idx = (int(np.mean(roi_idxs[0])),
                   int(np.mean(roi_idxs[1])),
                   int(np.mean(roi_idxs[2])))
        else:
            print 'No nodules'
            idx = np.array(input.shape) / 2
    else:
        idx = idx.astype(int)
    if axis == 0:  # sax
        ax[0, 0].imshow(prediction[idx[0], :, :], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[idx[0], :, :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[idx[0], :, :], cmap=plt.cm.gray)
        ax[1, 1].imshow(lung_mask[idx[0], :, :], cmap=plt.cm.gray)
    if axis == 1:  # 2 lungs
        ax[0, 0].imshow(prediction[:, idx[1], :], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[:, idx[1], :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[:, idx[1], :], cmap=plt.cm.gray)
        ax[1, 1].imshow(lung_mask[:, idx[1], :], cmap=plt.cm.gray)
    if axis == 2:  # side view
        ax[0, 0].imshow(prediction[:, :, idx[2]], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[:, :, idx[2]], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[:, :, idx[2]], cmap=plt.cm.gray)
        ax[1, 1].imshow(lung_mask[:, :, idx[2]], cmap=plt.cm.gray)
    if img_dir is not None:
        fig.savefig(img_dir + '/%s-%s.png' % (pid, axis), bbox_inches='tight')
    else:
        plt.show()
    fig.clf()
    plt.close('all')


def plot_slice_3d_3_patch(input, mask, prediction, axis, pid, patch_size=64, img_dir=None, idx=None):
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
        sz, sy, sx = slice(idx[0], idx[0] + 1), slice(idx[1] - patch_size, idx[1] + patch_size), slice(
            idx[2] - patch_size, idx[2] + patch_size)
        ax[0, 0].imshow(prediction[sz, sy, sx], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[sz, sy, sx], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[sz, sy, sx], cmap=plt.cm.gray)
    if axis == 1:  # 2 lungs
        ax[0, 0].imshow(prediction[:, idx[1], :], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[:, idx[1], :], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[:, idx[1], :], cmap=plt.cm.gray)
    if axis == 2:  # side view
        ax[0, 0].imshow(prediction[:, :, idx[2]], cmap=plt.cm.gray)
        ax[1, 0].imshow(input[:, :, idx[2]], cmap=plt.cm.gray)
        ax[0, 1].imshow(mask[:, :, idx[2]], cmap=plt.cm.gray)
    if img_dir is not None:
        fig.savefig(img_dir + '/%s.png' % pid, bbox_inches='tight')
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


def plot_2d_animation(input, mask, predictions):
    rgb_image = np.concatenate((input, input, input), axis=0)
    mask = np.concatenate((np.zeros_like(input), mask, predictions), axis=0)
    # green = targets
    # blue = predictions
    # red = overlap

    idxs = np.where(mask > 0.3)
    rgb_image[idxs] = mask[idxs]

    rgb_image = np.rollaxis(rgb_image, axis=0, start=4)
    print rgb_image.shape

    def get_data_step(step):
        return rgb_image[step, :, :, :]

    fig = plt.figure()
    im = fig.gca().imshow(get_data_step(0))

    def init():
        im.set_data(get_data_step(0))
        return im,

    def animate(i):
        im.set_data(get_data_step(i))
        return im,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=rgb_image.shape[1],
                                   interval=20000 / rgb_image.shape[0],
                                   blit=True)

    def on_click(event):
        global anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    fig.canvas.mpl_connect('button_press_event', on_click)
    try:
        plt.show()
    except AttributeError:
        pass


def plot_learning_curves(train_losses, valid_losses, expid, img_dir):
    fig = plt.figure()
    x_range = np.arange(len(train_losses)) + 1

    plt.plot(x_range, train_losses)
    plt.plot(x_range, valid_losses)

    if img_dir is not None:
        fig.savefig(img_dir + '/%s.png' % expid, bbox_inches='tight')
        print 'Saved plot'
    else:
        plt.show()
    fig.clf()
    plt.close('all')
