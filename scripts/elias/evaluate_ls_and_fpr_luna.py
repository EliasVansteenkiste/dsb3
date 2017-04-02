"""
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import theano
import numpy as np
import theano.tensor as T
import os
import sys
import cPickle as pickle
import string
import lasagne
import time
import csv
from itertools import product
from collections import defaultdict
from scipy.stats import rankdata


sys.path.append(".")
from utils import LOGS_PATH, MODEL_PATH, MODEL_PREDICTIONS_PATH, paths
from utils.configuration import config, get_configuration, set_configuration
from interfaces.data_loader import TRAIN, VALIDATION, TEST, INPUT
from application.luna import LunaDataLoader

from configurations.elias import segnet1
from extract_nodules import check_in_mask
from apply_extractor import normHU2HU


import numpy as np
import skimage.measure
import skimage.segmentation
import skimage.morphology
import skimage.filters

import scipy.ndimage
import decimal


MAX_HU = 400.
MIN_HU = -1000.



def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)

def pixelnormHU_01(x):
    x = (x - MIN_HU) / (MAX_HU - MIN_HU)
    return np.clip(x, 0., 1., out=x)

def read_dict(prediction_folder, patient_id):
    return pickle.load( open( prediction_folder+patient_id+'.pkl', 'rb' ) )
    
def check_nodules(set, roi_config, roi_ls_fpr_folder):
    tags = ["luna:patient_id", "luna:labels", "luna:origin", "luna:pixelspacing", "luna:labels"]

    print 'checking nodules for set', set, 
    set_indices = roi_config.data_loader.indices[set]
    print 'no_samples', len(set_indices)
    n_nodules, n_found_rois, n_found_in_masks, n_regions, n_regions_in_mask = 0, 0, 0, 0, 0
    all_fpr_ps = []
    tp_fpr_ps = []
    tp_fpr_rank = []
    tpls_fpr_ps = []
    tpls_fpr_rank = []
    
    for _i, sample_id in enumerate(set_indices):
        # if _i == 10:
        #     break
        print "sample_id", sample_id, _i+1, "/", len(set_indices), "in", set
        data = roi_config.data_loader.load_sample(sample_id, tags,{})
        
        patient_id = data["input"]["luna:patient_id"]
        nodules = data["input"]["luna:labels"]
        origin = data["input"]["luna:origin"]
        spacing = data["input"]["luna:pixelspacing"]
        nodules = data["input"]["luna:labels"]

        dict = read_dict(roi_ls_fpr_folder, patient_id)
        
        rois = dict['rois']
        n_regions += len(rois)
        
        rois_in_mask = dict['in_mask']
        n_regions_in_mask += np.sum(rois_in_mask)

        fpr_p = dict['fpr_p']
        all_fpr_ps.append(fpr_p)

        fpr_lung_p = np.copy(fpr_p)
        fpr_lung_p[rois_in_mask==0] = 0

        #save rois that are above the 
        rank_rois = len(fpr_p)-rankdata(fpr_p).astype(int)
        rank_rois_lung = len(fpr_lung_p)-rankdata(fpr_lung_p).astype(int)

        max_dim = 0 
        for nidx, nodule in enumerate(nodules):
            n_nodules += 1
            print 'nodule orig coos', nodule
            n = LunaDataLoader.world_to_voxel_coordinates(nodule[:3], origin, spacing)
            print n
            diameter_in_mm =  nodule[3]
            nodule = n[:3]

            # apply spacing
            nodule = nodule*spacing
            print 'after spacing', nodule

            # Find the closest region of interest
            closest_roi = None
            min_distance = 99999999.
            min_idx = 99999999
            for roi_idx, roi in enumerate(rois):
                md = max(roi)
                if md > max_dim:
                    max_dim = md
                distance = sum((roi-nodule)**2)**(0.5)
                if distance < min_distance:
                    min_distance = distance
                    closest_roi = roi
                    min_idx = roi_idx

            print 'max_dim', max_dim
            print 'n', n
            print 'closest_roi', closest_roi
            print 'min_distance', min_distance
            print 'diameter', diameter_in_mm

            found = False
            if min_distance < diameter_in_mm:
                n_found_rois += 1
                print 'found', n_found_rois, '/', n_nodules
                found = True
                tp_fpr_ps.append(fpr_p[min_idx])
                tp_fpr_rank.append(rank_rois[min_idx])

             # Find the closest roi in lung mask
            closest_roi = None
            min_distance = 99999999.
            min_idx = 99999999
            for roi_idx, roi in enumerate(rois):
                if rois_in_mask[roi_idx]:
                    distance = sum((roi-nodule)**2)**(0.5)
                    if distance < min_distance:
                        min_distance = distance
                        closest_roi = roi
                        min_idx = roi_idx

            print 'closest_roi in mask', closest_roi
            print 'min_distance in mask', min_distance

            if min_distance < diameter_in_mm:
                n_found_in_masks += 1
                print 'found in mask', n_found_in_masks, '/', n_nodules
                tpls_fpr_ps.append(fpr_p[min_idx])
                tpls_fpr_rank.append(rank_rois_lung[min_idx])
            # elif found:
            #     plot_masks(closest_roi/spacing,volume,sample_id, nidx)


    print 'n_regions', n_regions
    print 'n_regions in lung masks', n_regions_in_mask     

    tp_fpr_ps = np.hstack(tp_fpr_ps)
    tpls_fpr_ps = np.hstack(tpls_fpr_ps)
    all_fpr_ps = np.hstack(all_fpr_ps) 
    tp_fpr_rank = np.hstack(tp_fpr_rank) 
    tpls_fpr_rank = np.hstack(tpls_fpr_rank)

    print '============ Sweep p cutoff ================'

    for pcutoff in [0.01, 0.015, 0.02, 0.025,  0.03,  0.035,  0.04, 0.05, 0.1, 0.2, 0.5]:
       print 'cutoff', pcutoff
       print 'tp_fpr_ps', np.sum(tp_fpr_ps > pcutoff), '/', len(tp_fpr_ps)
       print 'tpls_fpr_ps', np.sum(tpls_fpr_ps > pcutoff), '/', len(tpls_fpr_ps)
       print 'all_fpr_ps', np.sum(all_fpr_ps > pcutoff), '/', len(all_fpr_ps)


    print '============ Sweep Top x ================'

    for topx in [4, 6, 8, 10, 12, 14, 16]:
        print 'top', topx
        print 'tp', np.sum(tp_fpr_rank < topx), '/', len(tp_fpr_rank)
        print 'tp_ls', np.sum(tpls_fpr_rank < topx), '/', len(tpls_fpr_rank)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help='configuration to run',)
    args = parser.parse_args()
    set_configuration(args.config)

    roi_config, roi_config_name = get_configuration(config.roi_model_config)
    roi_config.data_loader.prepare()

    roi_ls_fpr_folder = paths.MODEL_PREDICTIONS_PATH + '/' + config.output_name  + '/'
    if not os.path.isdir(roi_ls_fpr_folder):
        raise
    
    for set in [VALIDATION]:
        check_nodules(set, roi_config, roi_ls_fpr_folder)

