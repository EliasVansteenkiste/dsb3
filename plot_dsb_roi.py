import cPickle as pickle
import string
import sys
import time
from itertools import izip
import lasagne as nn
import numpy as np
import theano
from datetime import datetime, timedelta
import utils
import logger
import theano.tensor as T
import buffering
from configuration import config, set_configuration
import pathfinder
import utils_plots

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_class_dsb', config_name)

predictions_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

train_data_iterator = config().train_data_iterator
valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples
print 'n chunks per epoch', config().nchunks_per_epoch

# use buffering.buffered_gen_threaded()
for (x_chunk_train, y_chunk_train, id_train) in valid_data_iterator.generate():
    print id_train
    print x_chunk_train.shape

    for i in xrange(x_chunk_train.shape[0]):
        pid = id_train[i]
        for j in xrange(x_chunk_train.shape[1]):
            utils_plots.plot_slice_3d_3axis(input=x_chunk_train[i, j, 0],
                                            pid='-'.join([str(pid), str(j)]),
                                            img_dir=outputs_path,
                                            idx=np.array(x_chunk_train[i, j, 0].shape) / 2)
