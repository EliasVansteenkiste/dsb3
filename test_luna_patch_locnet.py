import string
import sys
import lasagne as nn
import numpy as np
import theano
import buffering
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_3
import utils_lung

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_luna_patch', config_name)

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

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

givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared

# theano functions
iter_get_predictions = theano.function([], nn.layers.get_output(model.l_out), givens=givens_valid)
valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n validation: %d' % valid_data_iterator.nsamples

valid_losses_dice = []
valid_losses_ce = []
for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(valid_data_iterator.generate())):
    # load chunk to GPU
    x_shared.set_value(x_chunk)
    print 'loaded chunk'
    predictions = iter_get_predictions()
    targets = y_chunk
    inputs = x_chunk
    print 'dice'

    z = y_chunk[0, 0, :]
    y = y_chunk[0, 1, :]
    x = y_chunk[0, 2, :]
    targets = z[:, None, None] * y[None, :, None] * x[None, None, :]

    z = predictions[0, 0, :]
    y = predictions[0, 1, :]
    x = predictions[0, 2, :]
    pp = z[:, None, None] * y[None, :, None] * x[None, None, :]

    for k in xrange(predictions.shape[0]):
        plot_slice_3d_3(input=inputs[k, 0], mask=targets, prediction=pp,
                        axis=0, pid='-'.join([str(n), str(k), str(id_chunk[k])]),
                        img_dir=outputs_path)

print 'Dice index validation loss', np.mean(valid_losses_dice)
print 'CE validation loss', np.mean(valid_losses_ce)
