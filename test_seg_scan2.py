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

predictions_path = '/mnt/storage/metadata/dsb3/model-predictions/ikorshun/luna_s_p5_pixelnorm'

for n, (x, y, id, annotations, transform_matrices) in enumerate(valid_data_iterator.generate()):
    pid = id[0]
    annotations = annotations[0]
    tf_matrix = transform_matrices[0]

    blobs = utils.load_np(predictions_path + '/blob_%s.npy' % pid)
    blobs_original_voxel_coords = []
    for j in xrange(blobs.shape[0]):
        blob_j = np.append(blobs[j, :3], [1])
        blobs_original_voxel_coords.append(tf_matrix.dot(blob_j))
    blobs_original_voxel_coords = np.asarray(blobs_original_voxel_coords)
    pid2blobs[pid] = np.copy(blobs_original_voxel_coords)

utils.save_pkl(pid2blobs, path=predictions_path + 'candidates.pkl')
print 'Candidates saved'
