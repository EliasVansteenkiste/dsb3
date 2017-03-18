import sys
import lasagne as nn
import numpy as np
import theano
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_4
import theano.tensor as T
import blobs_detection
import logger
import time
import multiprocessing as mp
import buffering

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_seg_scan', config_name)

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s.log' % config_name)
sys.stderr = sys.stdout

data_iterator = config().data_iterator

print
print 'Data'
print 'n samples: %d' % data_iterator.nsamples

start_time = time.time()
n_pos = 0
tp = 0
for n, (x, y, lung_mask, annotations, tf_matrix, pid) in enumerate(data_iterator.generate()):
    print '-------------------------------------'
    print n, pid
    n_pos += annotations.shape[0]
    n_pid_tp = 0
    annotations = np.int32(annotations)
    for i in xrange(annotations.shape[0]):
        if lung_mask[0, 0, annotations[i, 0], annotations[i, 1], annotations[i, 2]] == 1:
            n_pid_tp += 1
    tp += n_pid_tp
    print annotations.shape[0], n_pid_tp
    if annotations.shape[0] > n_pid_tp:
        print '----HERE-----!!!!!'

print 'total', n_pos
print 'detected', tp
