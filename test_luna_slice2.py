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
import theano.tensor as T

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration(config_name)

expid = utils.generate_expid(config_name)
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

valid_loss = config().build_objective(model, deterministic=True)

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))

idx_z = T.lscalar('idx_z')
idx_y = T.lscalar('idx_y')
idx_x = T.lscalar('idx_x')
givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared[:, :,
                                     idx_z * 32:(idx_z * 32) + 64,
                                     idx_y * 32:(idx_y * 32) + 64,
                                     idx_x * 32:(idx_x * 32) + 64]

# theano functions
iter_get_predictions = theano.function([idx_z, idx_x, idx_y],
                                       nn.layers.get_output(model.l_out),
                                       givens=givens_valid,
                                       on_unused_input='ignore')
valid_data_iterator = config().valid_data_iterator

print
print 'Data'
print 'n validation: %d' % valid_data_iterator.nsamples

valid_losses = []
for n, (x_chunk, y_chunk, id_chunk) in enumerate(
        buffering.buffered_gen_threaded(valid_data_iterator.generate())):

    pp_out = np.zeros_like(x_chunk)
    x_shared.set_value(x_chunk)

    for iz in xrange(10):
        for iy in xrange(10):
            for ix in xrange(10):
                print iz, iy, ix
                # We should do something about the borders !!!!!!!!
                pp = iter_get_predictions(iz, iy, ix)
                print pp.shape
                print pp[0, 0, 16:48, 16:48, 16:48].shape
                print pp_out.shape
                pp_out[0, 0,
                16 + iz * 32:16 + (iz + 1) * 32,
                16 + iy * 32:16 + (iy + 1) * 32,
                16 + ix * 32:16 + (ix + 1) * 32] = pp[0, 0, 16:48, 16:48, 16:48]

                # plot_slice_3d_3(
                #     input=x_chunk_train[0, 0, iz * 64:(iz + 1) * 64, iy * 64:(iy + 1) * 64, ix * 64:(ix + 1) * 64],
                #     mask=y_chunk_train[0, 0, iz * 64:(iz + 1) * 64, iy * 64:(iy + 1) * 64, ix * 64:(ix + 1) * 64],
                #     prediction=pp[0, 0],
                #     axis=0, pid='-'.join([str(n), str(id_train[0])]),
                #     img_dir=outputs_path)
    # print 'plotting'
    # plot_slice_3d_3(input=x_chunk_train[0, 0], mask=y_chunk_train[0, 0], prediction=pp_out[0, 0],
    #                 axis=0, pid='-'.join([str(n), str(id_train[0])]),
    #                 img_dir=outputs_path)
    utils.save_pkl(pp_out, outputs_path + '/el_pred_' + id_chunk[0] + '.pkl')
    utils.save_pkl(x_chunk, outputs_path + '/el_in_' + id_chunk[0] + '.pkl')
    utils.save_pkl(y_chunk, outputs_path + '/el_tgt_' + id_chunk[0] + '.pkl')
    print 'Saved', id_chunk[0]
