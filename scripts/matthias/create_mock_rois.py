import numpy as np
import cPickle
from glob import glob
import sys

sys.path.append('.')

import utils.paths
from utils.paths import MODEL_PREDICTIONS_PATH


def create_mock_rois(roi_config):
    
    if not isinstance(roi_config, str): roi_config = roi_config.__name__
    #load rois
    rootdir = MODEL_PREDICTIONS_PATH + roi_config

    import os
    target_root_dir=MODEL_PREDICTIONS_PATH + '/mock_rois/'
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)
        
    
    #load data
    for path in glob(rootdir+"/*"):
        
        patient_id = path.split("/")[-1][:-4]

        print "reading rois for patient {}".format(patient_id)
        with open(path, "rb") as f: patient_rois = cPickle.load(f)
        
        rois_with_probs=np.empty((patient_rois.shape[0],patient_rois.shape[1]+1))

        for (roiidx,roi) in enumerate(patient_rois):
            prob=np.random.uniform()
            rois_with_probs[roiidx]=np.append(roi,prob)

        print "writing mock rois for patient {}".format(patient_id)

        patient_mock_path=target_root_dir+patient_id+".pkl"
        
        with open(patient_mock_path, "wb") as f: cPickle.dump(rois_with_probs,f)
        

if __name__ == '__main__':

    create_mock_rois("configurations.elias.roi_stage1_1")
