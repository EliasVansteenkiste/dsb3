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



train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

all_pids = train_pids + valid_pids

data_iterator = data_iterators.LunaSimpleDataGenerator(data_path=pathfinder.LUNA_DATA_PATH, patient_ids=all_pids)

histograms = {}
# bins = np.arange(-960,1700,40)
# avg_histogram = np.zeros((bins.shape[0]-1), dtype=np.int64)
# use buffering.buffered_gen_threaded()
for idx, (x, pid) in enumerate(data_iterator.generate()):
    print idx, 'pid', pid
    # if (idx == 10):
    #     break
    histograms[pid]= data_transforms.get_rescale_params_hist_eq(x)


#plot avg histogram
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, avg_histogram, align='center', width=width)
# plt.savefig('dsb_histogram_avg.jpg')
# plt.clf()


pickle.dump(histograms, open( "luna_rescale_params_hist_eq.pkl", "wb" ) )



