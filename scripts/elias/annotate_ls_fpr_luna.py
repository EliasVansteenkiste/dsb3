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
from utils.configuration import set_configuration, config, get_configuration, get_configuration_name
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

def ensure_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

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
    return pickle.load(open( prediction_folder+patient_id+'.pkl', 'rb' ))
    
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
    
def plot_masks(roi, ct_scan, sample_id, nidx):
    roi = np.array(roi).astype(int)
        
    #check if the candidate is in the lungs
    d0_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[roi[0],:,:]))
    d1_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[:,roi[1],:]))
    d2_segmentation_mask = segment_HU_scan_2d(pixelnormHU_01(ct_scan[:,:,roi[2]]))

    center = np.round(roi).astype(int)
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

def load_and_build_model(fpr_config, model_params_path):
    if os.path.isfile(model_params_path):
        print "build model"
        interface_layers = fpr_config.build_model()
        output_layers = interface_layers["outputs"]
        input_layers = interface_layers["inputs"]
        # merge all output layers into a fictional dummy layer which is not actually used
        top_layer = lasagne.layers.MergeLayer(
            incomings=output_layers.values()
        )

        print "Load model parameters"
        metadata = np.load(model_params_path)
        lasagne.layers.set_all_param_values(top_layer, metadata['param_values'])

        # Create the Theano variables necessary to interface with the models
        # the input:
        x_shared = {
            key: lasagne.utils.shared_empty(dim=len(l_in.output_shape), dtype='float32')
            for (key, l_in) in input_layers.iteritems()
        }


        network_outputs = [
            lasagne.layers.helper.get_output(network_output_layer, deterministic=True)
            for network_output_layer in output_layers.values()
        ]

        givens_valid = {}
        for (key, l_in) in input_layers.iteritems():
            givens_valid[l_in.input_var] = x_shared[key]

        iter_predict  = theano.function([], network_outputs, #+ theano_printer.get_the_stuff_to_print(),
                                        givens=givens_valid, on_unused_input="ignore")

        return x_shared, iter_predict
    else:
        raise 'model parameter path does not exist'

def get_prediction(x, x_shared, iter_predict):
    assert(len(x_shared.values())==1)
    x_shared.values()[0].set_value(x)
    predictions = iter_predict()
    return predictions        

#@profile
def check_nodules(set, roi_config, fpr_config, iter_predict, x_shared, prediction_folder, output_folder, plot=False):
    tags = ["luna:patient_id", "luna:origin", "luna:pixelspacing", "luna:labels", "luna:shape", "luna:3d"]

    print 'checking nodules for set', set, 
    set_indices = roi_config.data_loader.indices[set]
    print 'no_samples', len(set_indices)
    n_nodules, n_found_rois, n_found_in_masks, n_regions, n_regions_in_mask = 0, 0, 0, 0, 0


    rd = np.array(fpr_config.nn_input_shape)/2
    pw = rd[0]    
    for _i, sample_id in enumerate(set_indices):
        # if _i == 10:
        #     break
        print "sample_id", sample_id, _i+1, "/", len(set_indices), "in", set
        data = roi_config.data_loader.load_sample(sample_id, tags,{})
        
        patient_id = data["input"]["luna:patient_id"]
        origin = data["input"]["luna:origin"]
        spacing = data["input"]["luna:pixelspacing"]
        nodules = data["input"]["luna:labels"]
        shape = data["input"]["luna:shape"]
        volume = data["input"]["luna:3d"]
        pvolume = np.pad(volume,pw,'constant')

        rois = read_rois(prediction_folder, patient_id)
        n_regions += len(rois)
        
        rois_in_mask = np.zeros((len(rois)))
        for idx, roi in enumerate(rois):
            if in_mask(roi/spacing,volume):
                rois_in_mask[idx] = 1
        n_regions_in_mask += np.sum(rois_in_mask)

        fpr_p = np.zeros((len(rois)),dtype=np.float32)
        batch_size = 32
        for i in range(0,len(rois),batch_size):
            brois = rois[i:i+batch_size]
            brois = np.round(brois/spacing).astype(int)
            patches = []
            for center in brois:
                patch =  pvolume[pw+center[0]-rd[0]:pw+center[0]+rd[0], \
                            pw+center[1]-rd[1]:pw+center[1]+rd[1], \
                            pw+center[2]-rd[2]:pw+center[2]+rd[2]]
                patches.append(patch)    
            patches = np.array(patches)
            ypred = get_prediction(patches, x_shared, iter_predict)
            fpr_p[i:i+batch_size] = ypred[0][:,1]

        d_out = {'rois':rois, 'in_mask':rois_in_mask, 'fpr_p':fpr_p}

        pickle.dump(d_out, open(output_folder+patient_id+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help='configuration to run',)
    args = parser.parse_args()
    set_configuration(args.config)

    roi_config, roi_config_name = get_configuration(config.roi_model_config)
    roi_config.data_loader.prepare()

    fpr_config, fpr_config_name = get_configuration(config.fpr_model_config)

    prediction_folder = paths.MODEL_PREDICTIONS_PATH + '/' + config.prediction_loc  + '/'
    
    output_folder = paths.MODEL_PREDICTIONS_PATH + '/' + config.output_name  + '/'
    ensure_dir(output_folder)

    x_shared, iter_predict = load_and_build_model(fpr_config, config.saved_model_loc)
    
    for set in [VALIDATION]:
        check_nodules(set, roi_config, fpr_config, iter_predict, x_shared, prediction_folder, output_folder)

