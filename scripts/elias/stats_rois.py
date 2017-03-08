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


sys.path.append(".")
from utils import LOGS_PATH, MODEL_PATH, MODEL_PREDICTIONS_PATH, paths
from utils.configuration import set_configuration, config, get_configuration_name
from interfaces.data_loader import TRAIN, VALIDATION, TEST, INPUT
from application.luna import LunaDataLoader

from configurations.elias import segnet1



def read_rois(prediction_folder, patient_id):
    return pickle.load( open( prediction_folder+patient_id+'.pkl', 'rb' ) )
    


def check_nodules(set, plot=False):
    
    prediction_config = "configurations.elias.roi_luna_1"
    prediction_folder = paths.MODEL_PREDICTIONS_PATH + '/' + prediction_config  + '/'
    
    tags = ["luna:patient_id", "luna:origin", "luna:pixelspacing", "luna:labels", "luna:shape", "luna:3d"]

    print 'checking nodules for set', set, 
    set_indices = config.data_loader.indices[set]
    print 'no_samples', len(set_indices)
    n_nodules, n_found, n_regions = 0, 0, 0
    for _i, sample_id in enumerate(set_indices):
        print "sample_id", sample_id, _i+1, "/", len(set_indices), "in", set
        data = config.data_loader.load_sample(sample_id, tags,{})
        
        patient_id = data["input"]["luna:patient_id"]
        origin = data["input"]["luna:origin"]
        spacing = data["input"]["luna:pixelspacing"]
        nodules = data["input"]["luna:labels"]
        shape = data["input"]["luna:shape"]
        volume = data["input"]["luna:3d"]

        print 'pixelspacing', spacing
        print 'shape', shape

        rois = read_rois(prediction_folder, patient_id)
        n_regions += len(rois)


        max_dim = 0 
        for nidx, nodule in enumerate(nodules):
            n_nodules += 1
            print 'nodule orig coos', nodule
            n = LunaDataLoader.world_to_voxel_coordinates(nodule[:3], origin, spacing)
            print n
            diameter_in_mm =  nodule[3]
            nodule = n[:3]

            if plot:
                center = np.round(nodule).astype(int)
                fig = plt.figure()

                ax1 = fig.add_subplot(1,3,1, adjustable='box', aspect=1.0)
                ax1.imshow(volume[center[0],:,:].transpose(), interpolation='none', cmap=plt.cm.gray)
                circ1 = plt.Circle((center[1],center[2]), 24, color='y', fill=False)
                ax1.add_patch(circ1)

                ax2 = fig.add_subplot(1,3,2, adjustable='box', aspect=1.0)
                ax2.imshow(volume[:,center[1],:].transpose(), interpolation='none', cmap=plt.cm.gray)
                circ2 = plt.Circle((center[0],center[2]), 24, color='y', fill=False)
                ax2.add_patch(circ2)

                ax3 = fig.add_subplot(1,3,3, adjustable='box', aspect=1.0)
                ax3.imshow(volume[:,:,center[2]].transpose(), interpolation='none', cmap=plt.cm.gray)
                circ3 = plt.Circle((center[0],center[1]), 24, color='y', fill=False)
                ax3.add_patch(circ3)

                plt.tight_layout()
                fig.savefig(str(sample_id)+'_'+str(nidx)+'.jpg')

            # apply spacing
            nodule = nodule*spacing
            print 'after spacing', nodule

            # Find the closest region of interest
            closest_roi = None
            min_distance = 99999999.
            
            for roi in rois:
                md = max(roi)
                if md > max_dim:
                    max_dim = md
                distance = sum((roi-nodule)**2)**(0.5)
                if distance < min_distance:
                    min_distance = distance
                    closest_roi = roi

            print 'max_dim', max_dim
            print 'n', n
            print 'closest_roi', closest_roi
            print 'min_distance', min_distance
            print 'diameter', diameter_in_mm

            if min_distance < diameter_in_mm:
                n_found += 1
                print 'found', n_found, '/', n_nodules

    print 'n_regions', n_regions            



if __name__ == "__main__":
    # model = segnet1
    set_configuration("configurations/elias/roi_luna_1.py")
    config.data_loader.prepare()

    for set in [VALIDATION]:
        check_nodules(set)
