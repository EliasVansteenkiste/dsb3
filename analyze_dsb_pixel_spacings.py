import cPickle as pickle
import string
import sys
import time
from itertools import izip
import numpy as np
from datetime import datetime, timedelta
import utils
import utils_lung
import logger
import buffering
import pathfinder
import data_iterators


candidates_config = 'dsb_c3_s5_p8a1'
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
candidates_path = predictions_dir + '/%s' % candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
all_pids = train_pids + valid_pids + test_pids

data_iterator = data_iterators.DSBPixelSpacingsGenerator(pathfinder.DATA_PATH, id2candidates_path, all_pids)


z = []
y = []
x = []
pixel_spacings = {}

# use buffering.buffered_gen_threaded()
for idx, (pid, pixel_spacing) in enumerate(data_iterator.generate()):
    print idx, pid, pixel_spacing
    z.append(pixel_spacing[0])
    y.append(pixel_spacing[1])
    x.append(pixel_spacing[2])
    pixel_spacings[pid] = pixel_spacing


utils.save_pkl(pixel_spacings, 'pixel_spacings_dsb.pkl')
print 'z', min(z), max(z)
print 'y', min(y), max(y)
print 'x', min(x), max(x)
