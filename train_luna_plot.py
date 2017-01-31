import string
import sys
import time
from itertools import izip
import lasagne as nn
import numpy as np
import theano
import theano.tensor as T
from datetime import datetime, timedelta
import buffering
import logger
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_3

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration(config_name)
expid = utils.generate_expid(config_name)
print
print "Experiment ID: %s" % expid
print

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = metadata_dir + '/%s.pkl' % expid

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s.log' % expid)
sys.stderr = sys.stdout

# predictions path
predictions_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/' + expid
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

train_loss = config().build_objective(model, deterministic=False)
valid_loss = config().build_objective(model, deterministic=True)

learning_rate_schedule = config().learning_rate_schedule
learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))
updates = config().build_updates(train_loss, model, learning_rate)

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
y_shared = nn.utils.shared_empty(dim=len(model.l_target.shape))

idx = T.lscalar('idx')
givens_train = {}
givens_train[model.l_in.input_var] = x_shared[idx * config().batch_size:(idx + 1) * config().batch_size]
givens_train[model.l_target.input_var] = y_shared[idx * config().batch_size:(idx + 1) * config().batch_size]

givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared
givens_valid[model.l_target.input_var] = y_shared

# theano functions
iter_train = theano.function([idx], train_loss, givens=givens_train, updates=updates)
iter_get_predictions = theano.function([idx], nn.layers.get_output(model.l_out), givens=givens_train,
                                       on_unused_input='ignore')
iter_get_targets = theano.function([idx], nn.layers.get_output(model.l_target), givens=givens_train,
                                   on_unused_input='ignore')
iter_get_inputs = theano.function([idx], nn.layers.get_output(model.l_in), givens=givens_train,
                                  on_unused_input='ignore')
iter_validate = theano.function([], valid_loss, givens=givens_valid)

if config().restart_from_save:
    print 'Load model parameters for resuming'
    resume_metadata = utils.load_pkl(config().restart_from_save)
    nn.layers.set_all_param_values(model.l_out, resume_metadata['param_values'])
    start_chunk_idx = resume_metadata['chunks_since_start'] + 1
    chunk_idxs = range(start_chunk_idx, config().max_nchunks)

    lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
    print '  setting learning rate to %.7f' % lr
    learning_rate.set_value(lr)
    losses_eval_train = resume_metadata['losses_eval_train']
    losses_eval_valid = resume_metadata['losses_eval_valid']
else:
    chunk_idxs = range(config().max_nchunks)
    losses_eval_train = []
    losses_eval_valid = []
    start_chunk_idx = 0

train_data_iterator = config().valid_data_iterator
# TODO: hack here!!!!!
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

# use buffering.buffered_gen_threaded()
for chunk_idx, (x_chunk_train, y_chunk_train, id_train) in izip(chunk_idxs, buffering.buffered_gen_threaded(
        train_data_iterator.generate())):
    if chunk_idx in learning_rate_schedule:
        lr = np.float32(learning_rate_schedule[chunk_idx])
        print '  setting learning rate to %.7f' % lr
        print
        learning_rate.set_value(lr)

    # load chunk to GPU
    x_shared.set_value(x_chunk_train)
    y_shared.set_value(y_chunk_train)

    # make nbatches_chunk iterations
    for b in xrange(config().nbatches_chunk):
        if np.sum(y_chunk_train) > 0:
            loss = iter_train(b)
        else:
            print 'SKIP THIS!!'
        print chunk_idx, loss
        tmp_losses_train.append(loss)

        pp = iter_get_predictions(b)
        tt = iter_get_targets(b)
        ii = iter_get_inputs(b)
        for k in xrange(pp.shape[0]):
            try:
                plot_slice_3d_3(input=ii[k, 0], mask=tt[k, 0], prediction=pp[k, 0],
                                axis=0, pid='-'.join([str(chunk_idx), str(k), str(id_train[k])]),
                                img_dir=outputs_path)
                print 'Saved'
            except:
                print np.sum(tt[k, 0])
                print 'AAAAAAAAAAAAAAAA'
