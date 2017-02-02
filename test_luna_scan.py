import string
import sys
import lasagne as nn
import numpy as np
import theano
import buffering
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_3, plot_slice_3d_3_patch
import theano.tensor as T

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: train.py <configuration_name>")

config_name = sys.argv[1]
set_configuration(config_name)

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s-%s' % (expid, 'scan')
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

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
idx_z = T.lscalar('idx_z')
idx_y = T.lscalar('idx_y')
idx_x = T.lscalar('idx_x')

fs = config().p_transform['patch_size'][0]
stride = fs / 2
n_windows = (config().p_transform_scan['patch_size'][0] - stride) / stride  # TODO wrong

givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared[:, :,
                                     idx_z * stride:(idx_z * stride) + fs,
                                     idx_y * stride:(idx_y * stride) + fs,
                                     idx_x * stride:(idx_x * stride) + fs]

get_predictions_patch = theano.function([idx_z, idx_y, idx_x],
                                        nn.layers.get_output(model.l_out),
                                        givens=givens_valid,
                                        on_unused_input='ignore')

valid_data_iterator = config().valid_data_iterator_scan

print
print 'Data'
print 'n validation: %d' % valid_data_iterator.nsamples

for n, (x_chunk, y_chunk, id_chunk) in enumerate(valid_data_iterator.generate()):

    predictions_scan = np.zeros_like(x_chunk)
    x_shared.set_value(x_chunk)

    for iz in xrange(n_windows):
        for iy in xrange(n_windows):
            for ix in xrange(n_windows):
                print iz, iy, ix
                predictions_patch = get_predictions_patch(iz, iy, ix)
                predictions_scan[0, 0,
                iz * stride:iz * stride + fs,
                iy * stride:iy * stride + fs,
                ix * stride:ix * stride + fs] += predictions_patch[0, 0]

                mask = y_chunk[0, 0, iz * stride:(iz * stride) + fs,
                       iy * stride:(iy * stride) + fs,
                       ix * stride:(ix * stride) + fs]
                # if np.sum(mask) > 0:
                #     plot_slice_3d_3(
                #         input=ii[0, 0],
                #         mask=mask,
                #         prediction=pp[0, 0],
                #         axis=0, pid='-'.join([str(n), str(iz), str(iy), str(ix), str(id_chunk[0])]),
                #         img_dir=outputs_path)
    predictions_scan = np.clip(predictions_scan, 0, 1)

    plot_slice_3d_3(input=x_chunk[0, 0], mask=y_chunk[0, 0], prediction=predictions_scan[0, 0],
                    axis=0, pid='-'.join(['scan', str(n), str(id_chunk[0])]),
                    img_dir=outputs_path)
    print 'Saved plot'
    utils.save_pkl(predictions_scan, outputs_path + '/pred_' + id_chunk[0] + '.pkl')
    utils.save_pkl(x_chunk, outputs_path + '/in_' + id_chunk[0] + '.pkl')
    utils.save_pkl(y_chunk, outputs_path + '/tgt_' + id_chunk[0] + '.pkl')
    print 'Saved pkl', id_chunk[0]
