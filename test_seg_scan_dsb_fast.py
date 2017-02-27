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
import time

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_seg_scan', config_name)

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
utils.auto_make_dir(outputs_path)

# builds model and sets its parameters
model = config().build_model()

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))

fs = config().filter_size
stride = config().stride
n_windows = (config().p_transform['patch_size'][0] - fs) / stride + 1

givens = {}
idx = T.iscalar('idx')
batch_size = 1
givens[model.l_in.input_var] = x_shared

get_predictions_patch = theano.function([],
                                        nn.layers.get_output(model.l_out, deterministic=True),
                                        givens=givens,
                                        on_unused_input='ignore')

data_iterator = config().data_iterator

print
print 'Data'
print 'n samples: %d' % data_iterator.nsamples

n_blobs = 0
pid2blobs = {}
for n, (x, tf_matrix, pid) in enumerate(data_iterator.generate()):
    print '-------------------------------------'
    print n, pid
    start_time = time.time()

    x_patches = []
    for iz in xrange(n_windows):
        for iy in xrange(n_windows):
            for ix in xrange(n_windows):
                x_patches.append(x[:, :, iz * stride:(iz * stride) + fs,
                                 iy * stride:(iy * stride) + fs,
                                 ix * stride:(ix * stride) + fs])
    x_patches = np.concatenate(x_patches, axis=0)

    predictions_patches = []
    print x_patches.shape
    for i in xrange(int(np.ceil(1. * x_patches.shape[0] / batch_size))):
        print i
        x_shared.set_value(x_patches[i * batch_size:(i + 1) * batch_size])
        predictions_patches.append(get_predictions_patch())

    predictions_patches = np.concatenate(predictions_patches, axis=0)

    print predictions_patches.shape
    print 'time', (time.time() - start_time) / 60.
    i = 0
    predictions_scan = np.zeros((1, 1, n_windows * stride, n_windows * stride, n_windows * stride))
    for iz in xrange(n_windows):
        for iy in xrange(n_windows):
            for ix in xrange(n_windows):
                predictions_scan[0, 0,
                iz * stride:(iz + 1) * stride,
                iy * stride:(iy + 1) * stride,
                ix * stride:(ix + 1) * stride] = predictions_patches[i]
                i += 1

    if predictions_scan.shape != x.shape:
        pad_width = (np.asarray(x.shape) - np.asarray(predictions_scan.shape)) / 2
        pad_width = [(p, p) for p in pad_width]
        predictions_scan = np.pad(predictions_scan, pad_width=pad_width, mode='constant')

    plot_slice_3d_3(input=x[0, 0], mask=x[0, 0], prediction=predictions_scan[0, 0],
                    axis=0, pid='-'.join([str(n), str(pid)]),
                    img_dir=outputs_path, idx=np.array(x[0, 0].shape) / 2)
    print 'saved plot'

    print 'computing blobs'
    blobs = blobs_detection.blob_dog(predictions_scan[0, 0], min_sigma=1, max_sigma=15, threshold=0.1)

    n_blobs += len(blobs)
    print 'n_blobs detected', len(blobs)

    blobs_original_voxel_coords = []
    for j in xrange(blobs.shape[0]):
        blob_j = np.append(blobs[j, :3], [1])
        blob_j_original = tf_matrix.dot(blob_j)
        blobs_original_voxel_coords.append(blob_j_original)
    pid2blobs[pid] = np.asarray(blobs_original_voxel_coords)
    utils.save_pkl(pid2blobs, outputs_path + '/candidates.pkl')
