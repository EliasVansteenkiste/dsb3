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
test2_data_iterator = config().test_stage2_data_iterator

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples
print 'n chunks per epoch', config().nchunks_per_epoch

# use buffering.buffered_gen_threaded()
for (x_chunk_train, y_chunk_train, id_train) in test2_data_iterator.generate():
    print id_train
    print x_chunk_train.shape

    for i in xrange(x_chunk_train.shape[0]):
        pid = id_train[i]
        for j in xrange(x_chunk_train.shape[1]):
            utils_plots.plot_slice_3d_3axis(input=x_chunk_train[i, j, 0],
                                            pid='-'.join([str(pid), str(j)]),
                                            img_dir=outputs_path,
                                            idx=np.array(x_chunk_train[i, j, 0].shape) / 2)

ids = ['3b4c610fce3d4d723bc17986395af9ab',
       '3e4568aa1b37bd06f3917bc505ab6c2a',
       '401c2a2e7ff122ec5c558089d8ae3586',
       '419af46335739bb811e8bc97c3863836',
       '1f80571a52f38a5d9c029149612cb553',
       '3e4568aa1b37bd06f3917bc505ab6c2a',
       '3d8f006eeab0a4ea109ffe3901c7f695',
       '419af46335739bb811e8bc97c3863836',
       '417e3c40213fe0b8474b1ff74318f14c',
       '3d8f006eeab0a4ea109ffe3901c7f695']
