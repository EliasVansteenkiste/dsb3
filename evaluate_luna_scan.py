import sys
import os
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_3, plot_slice_3d_2
import data_transforms
import glob
import numpy as np
import utils_lung
from sandbox import blobs_detection

if len(sys.argv) < 2:
    sys.exit("Usage: evaluate_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_luna_scan', config_name)

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s/' % config_name

a_path = '/mnt/storage/metadata/dsb3/analysis/ikorshun/'

files = os.listdir(outputs_path)
x_files = sorted(glob.glob(outputs_path + '/in_*.npy'))
y_files = sorted(glob.glob(outputs_path + '/tgt_*.npy'))
pred_files = sorted(glob.glob(outputs_path + '/pred_*.npy'))

tp = 0
n_pos = 0
for xf, yf, pf in zip(x_files, y_files, pred_files):
    annotations_scan = utils.load_np(yf)
    x_scan = utils.load_np(xf)
    pred_scan = utils.load_np(pf)

    pid = utils_lung.luna_extract_pid(xf, '.npy').replace('in_', '')
    assert pid in yf
    assert pid in pf
    print pid

    # pred1_idxs = np.where(pred_scan > 0)
    print 'computing blobs'
    blobs = blobs_detection.blob_dog(pred_scan, min_sigma=1.2, max_sigma=35, threshold=0.1)
    # print blobs
    # print annotations_scan

    # mask = data_transforms.make_3d_mask_from_annotations(x_scan.shape, annotations_scan, shape='sphere')
    # for zyxd in annotations_scan:
    #     plot_slice_3d_3(x_scan, mask, pred_scan, 0, pid, idx=zyxd, img_dir=a_path)
    #     plot_slice_3d_3(x_scan, mask, pred_scan, 1, pid, idx=zyxd, img_dir=a_path)
    #     plot_slice_3d_3(x_scan, mask, pred_scan, 2, pid, idx=zyxd, img_dir=a_path)
    for zyxd in annotations_scan:
        n_pos += 1
        r = zyxd[-1] / 2.
        distance2 = ((zyxd[0] - blobs[:, 0]) ** 2
                     + (zyxd[1] - blobs[:, 1]) ** 2
                     + (zyxd[2] - blobs[:, 2]) ** 2)
        blob_idx = np.argmin(distance2)
        blob = blobs[blob_idx]
        print 'node', zyxd
        print 'closest blob', blob, blob_idx
        if distance2[blob_idx] < r ** 2:
            tp += 1
            print 'detected!!!'
        else:
            print 'not detected'

print tp
print n_pos
