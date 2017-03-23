import sys
import lasagne as nn
import numpy as np
import theano
import pathfinder
import utils
from configuration import config, set_configuration
from utils_plots import plot_slice_3d_3
import theano.tensor as T
import utils_lung
import blobs_detection
import logger
from collections import defaultdict

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: generate_features_dsb.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_gen_features', config_name)

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s.log' % config_name)
sys.stderr = sys.stdout

# builds model and sets its parameters
model = config().build_model()

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
givens_valid = {}
givens_valid[model.l_in.input_var] = x_shared

get_featuremap = theano.function([], nn.layers.get_output(model.l_out, deterministic=True),
                                        givens=givens_valid,
                                        on_unused_input='ignore')

data_iterator = config().data_iterator

print
print 'Data'
print 'n samples: %d' % data_iterator.nsamples

prev_pid = None
candidates = []
patients_count = 0
patch_size = 48
stride = 16
for n, (x, id) in enumerate(data_iterator.generate()):
    pid = id

    print(pid)
    print model.l_out.shape
    predictions = np.empty((x.shape[2]//patch_size, x.shape[3]//patch_size, x.shape[4]//patch_size,) + model.l_out.shape[1])
    print predictions.shape
    print 'x.shape', x.shape
    for i in np.arange(0,x.shape[2]-patch_size+1,patch_size):  
        for j in np.arange(0,x.shape[3]-patch_size+1,patch_size):  
            for k in np.arange(0,x.shape[4]-patch_size+1,patch_size):
                x_in = x[0,0,i:i+patch_size,j:j+patch_size,k:k+patch_size]
                print x_in.shape
                x_shared.set_value(x_in[None,None,:,:,:])
                fm = get_featuremap()
                predictions.append(fm)

    result = np.concatenate(predictions,axis=0)

    utils.save_pkl(result, outputs_path + '/%s.pkl' % pid)
