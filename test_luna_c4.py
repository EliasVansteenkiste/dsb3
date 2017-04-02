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
set_configuration('configs_fpred_patch', config_name)

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

xs_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_ins]
y_shared = nn.utils.shared_empty(dim=len(model.l_target.shape))

idx = T.lscalar('idx')
givens_train = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_train[l_in.input_var] = x[idx * config().batch_size:(idx + 1) * config().batch_size]
givens_train[model.l_target.input_var] = y_shared[idx * config().batch_size:(idx + 1) * config().batch_size]

givens_valid = {}
for l_in, x in izip(model.l_ins, xs_shared):
    givens_valid[l_in.input_var] = x
givens_valid[model.l_target.input_var] = y_shared

# theano functions
get_inputs = theano.function([], [nn.layers.get_output(l) for l in model.l_ins],
                             givens=givens_valid, on_unused_input='ignore')

data_iterator = config().valid_data_iterator

k = 0
for xs_chunk_train, y_chunk_train, id_train in buffering.buffered_gen_threaded(data_iterator.generate()):
    for x_shared, x in zip(xs_shared, xs_chunk_train):
        x_shared.set_value(x)
    y_shared.set_value(y_chunk_train)

    predictions = get_inputs()
    pid = id_train
    print pid

    for p in predictions:
        print p.shape
    for x in xs_chunk_train:
        print x.shape

    for j, (x_chunk, p_chunk) in enumerate(zip(xs_chunk_train, predictions)):
        for i in xrange(x_chunk.shape[0]):
            utils_plots.plot_slice_3d_3axis(input=p_chunk[i, 0],
                                            pid='-'.join(
                                                [str(k), str(pid), str(i), str(j), str(y_chunk_train[0])]),
                                            img_dir=outputs_path,
                                            idx=np.array(p_chunk[i, 0].shape) / 2)
            print 'saved'

    k += 1
