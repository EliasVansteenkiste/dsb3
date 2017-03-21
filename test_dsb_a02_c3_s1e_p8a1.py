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

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_class_dsb', config_name)

predictions_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)
metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s.log' % expid)
sys.stderr = sys.stdout

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

nn.layers.set_all_param_values(model.l_out, metadata['param_values'])

train_loss = config().build_objective(model, deterministic=False)
valid_loss = config().build_objective(model, deterministic=True)

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
y_shared = nn.utils.shared_empty(dim=len(model.l_target.shape))

givens_train = {}
givens_train[model.l_in.input_var] = x_shared
givens_train[model.l_target.input_var] = y_shared

givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared
givens_valid[model.l_target.input_var] = y_shared

# theano functions
iter_validate = theano.function([],
                                nn.layers.get_output(model.l_roi_p1, deterministic=True), givens=givens_valid,
                                on_unused_input='ignore')

train_data_iterator = config().train_data_iterator
valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n train: %d' % train_data_iterator.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples
print 'n chunks per epoch', config().nchunks_per_epoch

print
print 'Train model'
chunk_idx = 0
start_time = time.time()
prev_time = start_time
tmp_losses_train = []
losses_train_print = []

data_iter = config().valid_data_iterator
for x_chunk_train, y_chunk_train, id_train in buffering.buffered_gen_threaded(
        data_iter.generate()):
    pid = id_train[0]
    print pid, y_chunk_train

    # load chunk to GPU
    x_shared.set_value(x_chunk_train)
    y_shared.set_value(y_chunk_train)

    predictions = iter_validate()
    print predictions[0]
    print

    for i in xrange(x_chunk_train.shape[0]):
        for j in xrange(x_chunk_train.shape[1]):
            utils_plots.plot_slice_3d_3axis(input=x_chunk_train[i, j, 0],
                                            pid='-'.join(
                                                [str(pid), str(j), str(y_chunk_train[0]), str(predictions[0, j])]),
                                            img_dir=outputs_path,
                                            idx=np.array(x_chunk_train[i, j, 0].shape) / 2)
