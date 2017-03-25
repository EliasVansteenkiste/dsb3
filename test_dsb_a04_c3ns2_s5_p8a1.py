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
from collections import defaultdict
import utils_lung

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
config_name = 'dsb_a04_c3ns_s5_p8a1'
set_configuration('configs_class_dsb', config_name)

malignancy = np.array([0., 0.01, 0.15, 0.75, 1.])
malignancy = np.reshape(malignancy, (5, 1))

predictions_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

print 'Build model'
model = config().build_model()
all_layers = nn.layers.get_all_layers(model.l_out)
all_params = nn.layers.get_all_params(model.l_out)
num_params = nn.layers.count_params(model.l_out)
print '  number of parameters: %d' % num_params
print string.ljust('  layer output shapes:', 36),
print string.ljust('#params:', 10),
print 'output shape:'
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(num_param.__str__(), 10)
    print '    %s %s %s' % (name, num_param, layer.output_shape)

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
y_shared = nn.utils.shared_empty(dim=len(model.l_target.shape))

givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared
givens_valid[model.l_target.input_var] = y_shared

# theano functions
iter_validate = theano.function([],
                                nn.layers.get_output(model.l_out, deterministic=True),
                                givens=givens_valid,
                                on_unused_input='ignore')

train_data_iterator = config().train_data_iterator
valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples
print 'n chunks per epoch', config().nchunks_per_epoch

chunk_idx = 0
start_time = time.time()
prev_time = start_time
tmp_losses_train = []
losses_train_print = []

data_iter = config().test_data_iterator

pid2prediction, pid2label = {}, {}
pid2label = utils_lung.read_test_labels(pathfinder.TEST_LABELS_PATH)
for x_chunk_train, y_chunk_train, id_train in buffering.buffered_gen_threaded(
        data_iter.generate()):
    pid = id_train[0]
    # load chunk to GPU
    x_shared.set_value(x_chunk_train)
    y_shared.set_value(y_chunk_train)

    predictions = iter_validate()
    # print predictions
    p = 1. - np.prod(1. - predictions.dot(malignancy))
    pid2prediction[pid] = p
    # pid2label[pid] = y_chunk_train[0]
    print pid, pid2label[pid], p

print  utils_lung.evaluate_log_loss(pid2prediction, pid2label)
