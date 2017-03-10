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
import glob
import data_transforms

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test_luna_scan.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs_fpred_scan', config_name)

# predictions path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/%s' % config_name
outputs_img_path = predictions_dir + '/%s_img' % config_name
utils.auto_make_dir(outputs_img_path)

blob_files = sorted(glob.glob(outputs_path + '/*.pkl'))

p_transform = {'patch_size': (64, 64, 64),
               'mm_patch_size': (64, 64, 64),
               'pixel_spacing': (1., 1., 1.)
               }

for p in blob_files:
    pid = utils_lung.extract_pid_filename(p, '.pkl')
    blobs = utils.load_pkl(p)
    blobs = np.asarray(sorted(blobs, key=lambda x: x[-1], reverse=True))

    img, pixel_spacing = utils_lung.read_dicom_scan(pathfinder.DATA_PATH + '/' + pid)
    print pid
    for blob in blobs[:10]:
        patch_center = blob[:3]
        p1 = blob[-1]
        print p1
        x, _ = data_transforms.transform_patch3d(data=img,
                                                 luna_annotations=None,
                                                 patch_center=patch_center,
                                                 p_transform=p_transform,
                                                 pixel_spacing=pixel_spacing,
                                                 luna_origin=None,
                                                 world_coord_system=False)

        plot_slice_3d_3(input=x, mask=x, prediction=x,
                        axis=0, pid='-'.join([str(pid), str(p1)]),
                        img_dir=outputs_img_path, idx=np.array(x[0, 0].shape) / 2)
        # print 'saved'
