import cPickle as pickle
import string
import sys
import time
from itertools import izip
import numpy as np
from datetime import datetime, timedelta
import utils
import logger
import buffering
from configuration import config, set_configuration
import pathfinder
import utils_plots
import data_iterators
import data_transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


predictions_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + 'dsb_scan_histograms'
utils.auto_make_dir(outputs_path)

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
print 'n train', len(train_pids)
print 'n valid', len(valid_pids)
print 'n test', len(test_pids)

all_pids = train_pids + valid_pids + test_pids

data_iterator = data_iterators.DSBDataGenerator(data_path=pathfinder.DATA_PATH, patient_pids=all_pids)

histograms = {}
bins = np.arange(-960,1700,40)
# avg_histogram = np.zeros((bins.shape[0]-1), dtype=np.int64)
# use buffering.buffered_gen_threaded()
for idx, (x, pid) in enumerate(data_iterator.generate()):
    print idx, 'pid', pid
    histograms[pid]= data_transforms.get_rescale_params_hist_eq(x)



pickle.dump(histograms, open( "dsb_rescale_params_hist_eq.pkl", "wb" ) )
