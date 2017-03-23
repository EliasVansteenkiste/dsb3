import string
import sys
import lasagne as nn
import numpy as np
import theano
import utils
import logger
import buffering
from configuration import config, set_configuration
import pathfinder
import utils_lung
import os
import evaluate_submission

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test_class_dsb.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_class_dsb', config_name)

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
output_pkl_file = outputs_path + '/%s-%s' % (expid, 'public_LB.pkl')
output_csv_file = outputs_path + '/%s-%s' % (expid, 'public_LB.csv')

# if os.path.isfile(output_pkl_file):
#     pid2prediction = utils.load_pkl(output_pkl_file)
#     utils_lung.write_submission(pid2prediction, output_csv_file)
#     print 'saved csv'
#     print output_csv_file
#     sys.exit(0)

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

# theano functions
iter_test = theano.function([model.l_in.input_var], nn.layers.get_output(model.l_out, deterministic=True))

test_data_iterator = config().test_data_iterator

print
print 'Data'
print 'n test: %d' % test_data_iterator.nsamples

pid2prediction = {}
for i, (x_chunk_test, _, id_test) in enumerate(buffering.buffered_gen_threaded(
        test_data_iterator.generate())):
    predictions = iter_test(x_chunk_test)
    pid = id_test[0]
    pid2prediction[pid] = predictions.reshape((1,))[0]
    print i, pid, predictions

utils.save_pkl(pid2prediction, output_pkl_file)
print 'Saved predictions into pkl'

utils_lung.write_submission(pid2prediction, output_csv_file)
print 'Saved predictions into csv'
loss = evaluate_submission.leaderboard_performance(output_csv_file)
print loss
