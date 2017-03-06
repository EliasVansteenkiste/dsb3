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
from extract_nodules import check_in_mask
from apply_extractor import normHU2HU


import numpy as np
import skimage.measure
import skimage.segmentation
import skimage.morphology
import skimage.filters

import scipy.ndimage



MAX_HU = 400.
MIN_HU = -1000.

def pixelnormHU_01(x):
    x = (x - MIN_HU) / (MAX_HU - MIN_HU)
    return np.clip(x, 0., 1., out=x)

def segment_HU_scan_2d(x):
    mask = x

    # detect if additional borders were added.
    corner = 0
    for i in xrange(0,min(mask.shape[0],mask.shape[1])):
       if mask[i, i] == 0:
           corner = i
           break

    # select inner part
    part = mask[corner:-corner-1,corner:-corner-1]
    binary_part = part > 0.5

    # fill the body part
    filled = scipy.ndimage.binary_fill_holes(binary_part) # fill body
    selem = skimage.morphology.disk(5) # clear details outside of major body part
    filled_borders = skimage.morphology.erosion(filled, selem)
    filled_borders = 1 - filled_borders # flip mask

    # put mask back
    full_mask = np.ones((mask.shape[0],mask.shape[1]))
    full_mask[corner:-corner-1,corner:-corner-1] = filled_borders

    full_mask = np.asarray(full_mask,dtype=np.bool)

    # set outside to grey
    filled_borders = mask
    filled_borders[full_mask]=0.75


    # finally do the normal segmentation operations

    # change the disk value of this operation to make it less aggressive
    selem = skimage.morphology.disk(13)
    eroded = skimage.morphology.erosion(filled_borders, selem)

    selem = skimage.morphology.disk(2)
    closed = skimage.morphology.closing(eroded, selem)


    # threshold grey values
    t = 0.25
    mask = closed < t
    
    return mask


def read_rois(prediction_folder, patient_id):
    return pickle.load( open( prediction_folder+patient_id+'.pkl', 'rb' ) )
    

def in_mask(roi, ct_scan):
    roi = np.array(roi).astype(int)
        
    #check if the candidate is in the lungs
    d0_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[roi[0],:,:]))
    d1_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[:,roi[1],:]))
    d2_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[:,:,roi[2]]))

    #Construct 2D spherical mask
    r_mask=0
    dim_mask = 2*r_mask+1
    spherical_mask = np.zeros((dim_mask,dim_mask))
    for x in range(dim_mask):
        for y in range(dim_mask):
            dist = (x-r_mask)**2 + (y-r_mask)**2
            if dist <= (r_mask**2):
                spherical_mask[x,y] = 1


    in_mask_d0 = check_in_mask(d0_segmentation_mask, spherical_mask, r_mask, \
        roi[1]-r_mask, roi[1]+r_mask+1, roi[2]-r_mask, roi[2]+r_mask+1)

    in_mask_d1 = check_in_mask(d1_segmentation_mask, spherical_mask, r_mask, \
        roi[0]-r_mask, roi[0]+r_mask+1, roi[2]-r_mask, roi[2]+r_mask+1)

    in_mask_d2 = check_in_mask(d2_segmentation_mask, spherical_mask, r_mask, \
        roi[0]-r_mask, roi[0]+r_mask+1, roi[1]-r_mask, roi[1]+r_mask+1)

    """
    majority implementation: A shitty method to bypass the fact that lung segmentation 
    doesn't work when the lung touches the border of the image
    """
    return  (in_mask_d0 or in_mask_d1 or in_mask_d2)   
    

def plot_masks(roi, ct_scan, nidx):
    roi = np.array(roi).astype(int)
        
    #check if the candidate is in the lungs
    d0_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[roi[0],:,:]))
    d1_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[:,roi[1],:]))
    d2_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[:,:,roi[2]]))

    #Construct 2D spherical mask
    r_mask=0
    dim_mask = 2*r_mask+1
    spherical_mask = np.zeros((dim_mask,dim_mask))
    for x in range(dim_mask):
        for y in range(dim_mask):
            dist = (x-r_mask)**2 + (y-r_mask)**2
            if dist <= (r_mask**2):
                spherical_mask[x,y] = 1


    if plot:
        center = np.round(nodule).astype(int)
        fig = plt.figure()

        ax1 = fig.add_subplot(2,3,1, adjustable='box', aspect=1.0)
        ax1.imshow(ct_scan[roi[0],:,:].transpose(), interpolation='none', cmap=plt.cm.gray)
        circ1 = plt.Circle((roi[1],roi[2]), 24, color='y', fill=False)
        ax1.add_patch(circ1)

        ax2 = fig.add_subplot(2,3,2, adjustable='box', aspect=1.0)
        ax2.imshow(ct_scan[:,roi[1],:].transpose(), interpolation='none', cmap=plt.cm.gray)
        circ2 = plt.Circle((roi[0],roi[2]), 24, color='y', fill=False)
        ax2.add_patch(circ2)

        ax3 = fig.add_subplot(2,3,3, adjustable='box', aspect=1.0)
        ax3.imshow(ct_scan[:,:,roi[2]].transpose(), interpolation='none', cmap=plt.cm.gray)
        circ3 = plt.Circle((roi[0],roi[1]), 24, color='y', fill=False)
        ax3.add_patch(circ3)

        ax4 = fig.add_subplot(2,3,4, adjustable='box', aspect=1.0)
        ax4.imshow(d0_segmentation_mask.transpose(), interpolation='none', cmap=plt.cm.gray)
        circ4 = plt.Circle((roi[1],roi[2]), 24, color='y', fill=False)
        ax4.add_patch(circ4)

        ax5 = fig.add_subplot(2,3,5, adjustable='box', aspect=1.0)
        ax5.imshow(d1_segmentation_mask.transpose(), interpolation='none', cmap=plt.cm.gray)
        circ5 = plt.Circle((roi[0],roi[2]), 24, color='y', fill=False)
        ax5.add_patch(circ5)

        ax6 = fig.add_subplot(2,3,6, adjustable='box', aspect=1.0)
        ax6.imshow(d2_segmentation_mask.transpose(), interpolation='none', cmap=plt.cm.gray)
        circ6 = plt.Circle((roi[0],roi[1]), 24, color='y', fill=False)
        ax6.add_patch(circ6)

        plt.tight_layout()
        fig.savefig('temp/'+str(sample_id)+'_'+str(nidx)+'.jpg') 


def check_nodules(set, plot=False):
    
    prediction_config = "configurations.elias.roi_luna_1"
    prediction_folder = paths.MODEL_PREDICTIONS_PATH + '/' + prediction_config  + '/'
    
    tags = ["luna:patient_id", "luna:origin", "luna:pixelspacing", "luna:labels", "luna:shape", "luna:3d"]

    print 'checking nodules for set', set, 
    set_indices = config.data_loader.indices[set]
    print 'no_samples', len(set_indices)
    n_nodules, n_found_rois, n_found_in_masks, n_regions, n_regions_in_mask = 0, 0, 0, 0, 0
    for _i, sample_id in enumerate(set_indices):
        print "sample_id", sample_id, _i+1, "/", len(set_indices), "in", set
        data = config.data_loader.load_sample(sample_id, tags,{})
        
        patient_id = data["input"]["luna:patient_id"]
        origin = data["input"]["luna:origin"]
        spacing = data["input"]["luna:pixelspacing"]
        nodules = data["input"]["luna:labels"]
        shape = data["input"]["luna:shape"]
        volume = data["input"]["luna:3d"]

        rois = read_rois(prediction_folder, patient_id)
        n_regions += len(rois)
        
        rois_in_mask = []
        for roi in rois:
            if in_mask(roi/spacing,volume):
                rois_in_mask.append(roi)
        n_regions_in_mask += len(rois_in_mask)


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

            found = False
            if min_distance < diameter_in_mm:
                n_found_rois += 1
                print 'found', n_found_rois, '/', n_nodules
                found = True

             # Find the closest roi in lung mask
            closest_roi = None
            min_distance = 99999999.
            for roi in rois_in_mask:
                distance = sum((roi-nodule)**2)**(0.5)
                if distance < min_distance:
                    min_distance = distance
                    closest_roi = roi

            print 'closest_roi in mask', closest_roi
            print 'min_distance in mask', min_distance

            if min_distance < diameter_in_mm:
                n_found_in_masks += 1
                print 'found in mask', n_found_in_masks, '/', n_nodules
            elif found:
                plot_masks(roi/spacing,volume,nidx)

    print 'n_regions', n_regions
    print 'n_regions in lung masks', n_regions_in_mask           



if __name__ == "__main__":
    # model = segnet1
    set_configuration("configurations/elias/roi_luna_1.py")
    config.data_loader.prepare()

    for set in [VALIDATION]:
        check_nodules(set)

