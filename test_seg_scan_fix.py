import sys
import lasagne as nn
import numpy as np
import theano
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_3
import theano.tensor as T
import blobs_detection
import logger
import time
import multiprocessing as mp

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_seg_scan', config_name)

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n samples: %d' % valid_data_iterator.nsamples

start_time = time.time()
for n, (x, y, id, annotations, transform_matrices) in enumerate(valid_data_iterator.generate()):
    pid = id[0]
    print '-------------------------------------'
    print n, pid
    annotations = annotations[0]
    tf_matrix = transform_matrices[0]

    blobs_original = utils.load_pkl(outputs_path + '/%s.pkl' % pid)  # in original coordinates

    blobs = []
    for j in xrange(blobs_original.shape[0]):
        blob_j = np.append(blobs_original[j, :3], [1])
        blob_j_tf = np.linalg.inv(tf_matrix).dot(blob_j)
        blobs.append(blob_j_tf)

    blobs = np.asarray(blobs)

    print 'n_blobs detected', len(blobs)
    correct_blobs_idxs = []
    for zyxd in annotations:
        r = zyxd[-1] / 2.
        distance2 = ((zyxd[0] - blobs[:, 0]) ** 2
                     + (zyxd[1] - blobs[:, 1]) ** 2
                     + (zyxd[2] - blobs[:, 2]) ** 2)
        blob_idx = np.argmin(distance2)
        print 'node', zyxd
        print 'closest blob', blobs[blob_idx]
        if distance2[blob_idx] <= r ** 2:
            correct_blobs_idxs.append(blob_idx)
        else:
            print 'not detected !!!'

    blobs_original[:, -1] = 0
    for c in correct_blobs_idxs:
        blobs_original[c, -1] = 1

    utils.save_pkl(blobs_original, outputs_path + '/%s.pkl' % pid)
