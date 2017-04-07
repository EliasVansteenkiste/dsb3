import sys
import lasagne as nn
import numpy as np
import theano
import os

import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_3
import theano.tensor as T
import utils_lung
import blobs_detection
import logger
from collections import defaultdict

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_fpred_scan', config_name)

tta = sys.argv[2] if len(sys.argv) == 3 else None

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s.log' % config_name)
sys.stderr = sys.stdout

# builds model and sets its parameters
model = config().build_model()

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared

get_predictions_patch = theano.function([],
                                        nn.layers.get_output(model.l_out, deterministic=True),
                                        givens=givens_valid,
                                        on_unused_input='ignore')


if tta == 'tta':
    data_iterator = config().tt_data_iterator

    #existing_preds = [f.rsplit('.') for f in os.listdir(outputs_path)]
    #print existing_preds

    print
    print 'Data'
    print 'n samples: %d' % data_iterator.nsamples

    prev_pid = None
    candidates = []
    patients_count = 0
    for n, (x, candidate_zyxd, pid) in enumerate(data_iterator.generate()):

        if pid != prev_pid and prev_pid is not None:
            print patients_count, prev_pid, len(candidates)
            candidates = np.asarray(candidates)
            a = np.asarray(sorted(candidates, key=lambda x: x[-1], reverse=True))
            utils.save_pkl(a, outputs_path + '/%s.pkl' % prev_pid)
            print 'saved predictions'
            patients_count += 1
            candidates = []
        
        preds = []
        for bidx, pos in enumerate(range(0,x.shape[0],16)):
            print bidx
            x_batch = x[pos:pos+16]
            x_shared.set_value(x)
            predictions = get_predictions_patch()
            predictions = predictions[:, 1] if predictions.shape[-1] == 2 else predictions
            #print "predictions", predictions
            preds.append(predictions)
        
        preds = np.concatenate(preds)
        pred = np.average(preds)
        candidate_zyxdp = np.append(candidate_zyxd, [[pred]])
        candidates.append(candidate_zyxdp)

        prev_pid = pid

    # save the last one
    print patients_count, prev_pid, len(candidates)
    candidates = np.asarray(candidates)
    a = np.asarray(sorted(candidates, key=lambda x: x[-1], reverse=True))
    utils.save_pkl(a, outputs_path + '/%s.pkl' % prev_pid)
    print 'saved predictions'
else:
    data_iterator = config().data_iterator

    #existing_preds = [f.rsplit('.') for f in os.listdir(outputs_path)]
    #print existing_preds

    print
    print 'Data'
    print 'n samples: %d' % data_iterator.nsamples

    prev_pid = None
    candidates = []
    patients_count = 0
    for n, (x, candidate_zyxd, id) in enumerate(data_iterator.generate()):
        pid = id[0]

        if pid != prev_pid and prev_pid is not None:
            print patients_count, prev_pid, len(candidates)
            candidates = np.asarray(candidates)
            a = np.asarray(sorted(candidates, key=lambda x: x[-1], reverse=True))
            utils.save_pkl(a, outputs_path + '/%s.pkl' % prev_pid)
            print 'saved predictions'
            patients_count += 1
            candidates = []

        x_shared.set_value(x)
        predictions = get_predictions_patch()
        p1 = predictions[0][1]
        candidate_zyxdp = np.append(candidate_zyxd, [[p1]])
        candidates.append(candidate_zyxdp)

        prev_pid = pid

    # save the last one
    print patients_count, prev_pid, len(candidates)
    candidates = np.asarray(candidates)
    a = np.asarray(sorted(candidates, key=lambda x: x[-1], reverse=True))
    utils.save_pkl(a, outputs_path + '/%s.pkl' % prev_pid)
    print 'saved predictions'
