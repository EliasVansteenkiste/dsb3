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
import utils_lung
import data_iterators

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_fpred_scan', config_name)

predictions_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

# candidates after segmentations path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
segmentation_outputs_path = predictions_dir + '/%s' % config_name
id2candidates_path = utils_lung.get_candidates_paths(segmentation_outputs_path)

data_iterator = data_iterators.FixedCandidatesLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                transform_params=config().p_transform,
                                                                data_prep_fun=config().data_prep_function,
                                                                id2candidates_path=id2candidates_path)

print
print 'Data'
print 'n samples: %d' % data_iterator.nsamples

for (x_chunk_train, y_chunk_train, id_train) in data_iterator.generate():
    print id_train
    print x_chunk_train.shape

    for i in xrange(x_chunk_train.shape[0]):
        pid = id_train[i]
        for j in xrange(x_chunk_train.shape[1]):
            utils_plots.plot_slice_3d_3axis(input=x_chunk_train[i, j, 0],
                                            pid='-'.join([str(pid), str(j)]),
                                            img_dir=outputs_path,
                                            idx=np.array(x_chunk_train[i, j, 0].shape) / 2)
