import numpy as np
from glob import glob
import cPickle

from interfaces.data_loader import INPUT, OUTPUT
from augmentation_3d import sample_augmentation_parameters, Augment3D, augment_3d
from utils.paths import MODEL_PREDICTIONS_PATH

class AugmentROIProb(Augment3D):
    """
    Generates zero padded roi augmented patches
    """

    def __init__(self, max_rois, roi_config, *args, **kwargs):
        super(AugmentROIProb, self).__init__(*args, **kwargs)
        self.max_rois = max_rois
        
        if not isinstance(roi_config, str): roi_config = roi_config.__name__
        rootdir = MODEL_PREDICTIONS_PATH + roi_config

        self.rois = {}

        for path in glob(rootdir+"/*"):
            patient_id = path.split("/")[-1][:-4]
            with open(path, "rb") as f: patient_rois = cPickle.load(f)

            # add the elements to a list for easier sorting (numpy.sort does not take lambdas)
            rois_to_sort=[]
            for roi in patient_rois:
                rois_to_sort.append(roi)
                
            # sort them by probability
            sorted_rois=np.array(sorted(rois_to_sort,key=lambda coords: coords[-1],reverse=True))
            self.rois[patient_id]=sorted_rois

  

    @property
    def extra_input_tags_required(self):
        input_tags_extra = super(AugmentROIProb, self).extra_input_tags_required
        datasetnames = set()
        for tag in self.tags: datasetnames.add(tag.split(':')[0])
        input_tags_extra += [dsn + ":patient_id" for dsn in datasetnames]
        input_tags_extra += [dsn + ":3d" for dsn in datasetnames]
        return input_tags_extra

    def process(self, sample):
        augment_p = sample_augmentation_parameters(self.augmentation_params)

        for tag in self.tags:
            pixelspacingtag = tag.split(':')[0] + ":pixelspacing"
            assert pixelspacingtag in sample[INPUT], "tag %s not found" % pixelspacingtag
            spacing = sample[INPUT][pixelspacingtag]

            volume = sample[INPUT][tag]
            new_vol = np.zeros((self.max_rois,)+self.output_shape, volume.dtype)

            patient_id = sample[INPUT][tag.split(':')[0] + ":patient_id"]
            rois = self.rois[patient_id]
            #np.random.shuffle(rois)

            for i in range(min(len(rois), self.max_rois)):
                # mm to input space
                center_to_shift = -rois[i][:-1]/np.asarray(spacing, np.float)
                # print rois[i], center_to_shift

                new_vol[i] = augment_3d(
                    volume=volume,
                    pixel_spacing=spacing,
                    output_shape=self.output_shape,
                    norm_patch_shape=self.norm_patch_shape,
                    augment_p=augment_p,
                    center_to_shift=center_to_shift
                )

            sample[INPUT][tag] = new_vol # shape: (max_rois, X, Y, Z)


def create_mock_data(roi_config):
    
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
        
        print "patient_rois: {}".format(patient_rois)
        rois_with_probs=np.empty(patient_rois.shape)

        for (roiidx,roi) in enumerate(patient_rois):
            prob=np.random.uniform()
            rois_with_probs[roiidx]=np.append(roi,prob)

        print "rois_with_probs: {}".format(rois_with_probs)
            
        print "writing mock rois for patient {}".format(patient_id)

        patient_mock_path=target_root_dir+patient_id+".pkl"
        
        with open(patient_mock_path, "rb") as f: cPickle.dump(rois_with_probs,f)
        
