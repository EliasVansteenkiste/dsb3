import string
import sys
import lasagne as nn
import numpy as np
import theano
import buffering
import pathfinder
import utils
from configuration import config, set_configuration
import logger
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_size_patch', config_name)

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s-test.log' % expid)
sys.stderr = sys.stdout

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
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

nn.layers.set_all_param_values(model.l_out, metadata['param_values'])

valid_loss = config().build_objective(model, deterministic=True)

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
y_shared = nn.utils.shared_empty(dim=len(model.l_target.shape))

givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared
givens_valid[model.l_target.input_var] = y_shared

# theano functions
iter_get_predictions = theano.function([], [valid_loss, nn.layers.get_output(model.l_out, deterministic=True)],
                                       givens=givens_valid)
valid_data_iterator = config().valid_data_iterator
data_iter = config().valid_data_iterator

print
print 'Data'
print 'n valid: %d' % data_iter.nsamples
print 'n validation: %d' % valid_data_iterator.nsamples
print 'n chunks per epoch', config().nchunks_per_epoch

losses = []
predictions, labels = [], []
for i, (x_chunk_valid, y_chunk_valid, id_valid) in enumerate(buffering.buffered_gen_threaded(
        data_iter.generate())):
    pid = id_valid[0]
    # load chunk to GPU
    x_shared.set_value(x_chunk_valid)
    y_shared.set_value(y_chunk_valid)

    loss, probs = iter_get_predictions()
    losses.append(loss)
    print i, pid, y_chunk_valid
    print probs
    predictions.append(np.argmax(probs[0]))
    labels.append(int(y_chunk_valid[0, 0]))
    print predictions[-1], labels[-1]

print 'validation loss', np.mean(losses)
print confusion_matrix(labels, predictions)
