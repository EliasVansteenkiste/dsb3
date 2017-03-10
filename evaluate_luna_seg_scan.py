import glob
import os
import sys
import numpy as np
import blobs_detection
import pathfinder
import utils
import utils_lung
from configuration import set_configuration
import data_transforms

if len(sys.argv) < 2:
    sys.exit("Usage: evaluate_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_seg_scan', config_name)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name

blob_files = sorted(glob.glob(outputs_path + '/*.pkl'))
# print blob_files

pid2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

tp = 0
n_pos = 0
n_blobs = 0
for p in blob_files:
    pid = utils_lung.extract_pid_filename(p, '.pkl')
    blobs = utils.load_pkl(p)
    n_blobs += len(blobs)
    print pid
    print 'n_blobs', len(blobs)
    print 'tp / n pos ', int(np.sum(blobs[:, -1])), len(pid2annotations[pid])
    if int(np.sum(blobs[:, -1])) < len(pid2annotations[pid]):
        print '-------- HERE!!!!!! ------------'
    tp += np.sum(blobs[:, -1])
    print '====================================='

print 'n patients', len(blob_files)
print 'TP', tp
print 'n blobs', n_blobs
print n_pos
