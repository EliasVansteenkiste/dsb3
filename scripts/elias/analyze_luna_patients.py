import numpy as np
import pathfinder
import csv
import dicom
import os
import re
import numpy as np
import csv
import os
from PIL import Image
from collections import defaultdict
import time
import utils_lung

import utils
import cPickle as pickle


train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

n_positive, n_negative, n_only_negative, n_negative_pos = 0, 0, 0, 0


pos_ls_pos = []
pos_ls_neg = []
neg_ls = []

for pid in train_pids:
    if pid in id2positive_annotations:
        l_pos = len(id2positive_annotations[pid])
        l_neg = len(id2negative_annotations[pid])
        n_positive += l_pos
        n_negative += l_neg
        n_negative_pos += l_neg
        pos_ls_pos.append(l_pos)
        pos_ls_neg.append(l_neg)
        
    else:
        l_neg = len(id2negative_annotations[pid])
        n_negative += l_neg
        n_only_negative += l_neg
        neg_ls.append(l_neg)

print 'n positive', n_positive
print 'n negative', n_negative
print 'n only negative', n_only_negative
print 'n n_negative_pos', n_negative_pos
print 'Min en max voor het aantal per patient'
print 'pos_ls_pos', min(pos_ls_pos), max (pos_ls_pos)
print 'pos_ls_neg', min(pos_ls_neg), max(pos_ls_neg)
print 'neg_ls', min(neg_ls), max(neg_ls)


n_positive, n_negative, n_only_negative, n_negative_pos = 0, 0, 0, 0


pos_ls_pos = []
pos_ls_neg = []
neg_ls = []

for pid in valid_pids:
    if pid in id2positive_annotations:
        l_pos = len(id2positive_annotations[pid])
        l_neg = len(id2negative_annotations[pid])
        n_positive += l_pos
        n_negative += l_neg
        n_negative_pos += l_neg
        pos_ls_pos.append(l_pos)
        pos_ls_neg.append(l_neg)
        
    else:
        l_neg = len(id2negative_annotations[pid])
        n_negative += l_neg
        n_only_negative += l_neg
        neg_ls.append(l_neg)

print 'n positive', n_positive
print 'n negative', n_negative
print 'n only negative', n_only_negative
print 'n n_negative_pos', n_negative_pos
print 'Min en max voor het aantal per patient'
print 'pos_ls_pos', min(pos_ls_pos), max (pos_ls_pos)
print 'pos_ls_neg', min(pos_ls_neg), max(pos_ls_neg)
print 'neg_ls', min(neg_ls), max(neg_ls)




