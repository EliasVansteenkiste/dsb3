import sys
import numpy as np
import theano
import pathfinder
import utils
from configuration import set_configuration
import utils_lung

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_fpred_scan', config_name)

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
pid2candidates_path = utils_lung.get_candidates_paths(outputs_path)
pid2candidates = {}
for k, v in pid2candidates_path.iteritems():
    pid2candidates[k] = utils.load_pkl(v)

pid2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

n_top = 10
tp_top_n = 0
tp = 0
n_pos = 0
idx1 = []
for k in pid2candidates.iterkeys():
    print '----------------------------------------'
    print k
    n_true = len(pid2annotations[k])
    n_det_top = int(np.sum(pid2candidates[k][:n_top][:, 3]))
    n_det = int(np.sum(pid2candidates[k][:, 3]))
    i1 = np.where(pid2candidates[k][:, 3] == 1)[0]
    idx1.extend(pid2candidates[k][i1, -1])
    print 'n nodules', n_true
    print 'n nodules in top n', n_det_top
    print 'n nodules detected', n_det

    tp += n_det
    tp_top_n += n_det_top
    n_pos += n_true

print 'TP', tp
print 'TP in top n', tp_top_n
print 'n pos', n_pos
print np.sum(idx1) / n_pos
