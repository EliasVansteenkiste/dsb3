import os
import numpy as np
import data_transforms
import pathfinder
import utils
import utils_lung
from configuration import set_configuration, config
from utils_plots import plot_2d_animation, plot_slice_3d_3

set_configuration('test_config')


def test_luna3d():
    # path = '/mnt/sda3/data/kaggle-lung/lunapred/luna_scan_v3_dice-20170131-173443/'
    path = '/mnt/sda3/data/kaggle-lung/lunapred_el/luna_scan_v3_dice-20170201-231707/'
    files = os.listdir(path)
    print files
    x, y, p = [], [], []
    for f in files:
        if 'in' in f:
            x.append(f)
        elif 'tgt' in f:
            y.append(f)
        else:
            p.append(f)
    x = sorted(x)
    y = sorted(y)
    p = sorted(p)
    for xf, yf, pf in zip(x, y, p):
        x_batch = utils.load_pkl(path + xf)
        pred_batch = utils.load_pkl(path + pf)
        y_batch = utils.load_pkl(path + yf)
        print xf
        print yf
        print pf
        # plot_2d_animation(x_batch[0], y_batch[0], pred_batch[0])
        plot_slice_3d_3(x_batch[0,0],y_batch[0,0],pred_batch[0,0],0,'aa')



if __name__ == '__main__':
    test_luna3d()
