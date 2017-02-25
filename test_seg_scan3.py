import string
import sys
import lasagne as nn
import numpy as np
import theano
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_3
import theano.tensor as T
import utils_lung
import blobs_detection
import logger

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_seg_scan', config_name)

valid_data_iterator = config().valid_data_iterator
print
print 'Data'
print 'n validation: %d' % valid_data_iterator.nsamples

valid_losses_dice = []
n_pos = 0
tp = 0
n_blobs = 0
pid2blobs = {}

predictions_path = '/mnt/storage/metadata/dsb3/model-predictions/ikorshun/' + config_name
blobs = utils.load_pkl(predictions_path + '/candidates.pkl')
blobs2 = utils.load_pkl('/mnt/storage/metadata/dsb3/model-predictions/ikorshun/luna_s2_p5_pixelnorm/candidates.pkl')

for n, (x, y, id, annotations, transform_matrices) in enumerate(valid_data_iterator.generate()):
    pid = id[0]
    annotations = annotations[0]
    tf_matrix = transform_matrices[0]
    print blobs[pid].shape
    print 'p5', blobs2[pid].shape

    bb = blobs[pid]
    for i in xrange(bb.shape[0]):
        if bb[i, -1] == 1:
            print bb[i]
    print '---'
