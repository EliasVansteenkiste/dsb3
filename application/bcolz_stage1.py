import glob
import csv
import random
from os import path
import bcolz
bcolz.set_nthreads(2) # 2 instead of 12 by default
import sys
import cPickle
import gzip

import numpy as np

from interfaces.data_loader import StandardDataLoader, TRAINING, VALIDATION, TEST, INPUT, OUTPUT, TRAIN
from utils import paths


VALIDATION_SET_SIZE = 0.2
CORRUPT = {"b8bb02d229361a623a4dc57aa0e5c485"}


class BcolzStage1DataLoader(StandardDataLoader):

    datasets = [TRAIN, VALIDATION, TEST]

    def __init__(self, remove_corrupt=False, location=paths.BCOLZ_DATA_PATH, *args, **kwargs):
        super(BcolzStage1DataLoader, self).__init__(location=location, *args, **kwargs)
        self.remove_corrupt = remove_corrupt

    def prepare(self):
        # load only when not loaded yet
        if TRAINING in self.data and VALIDATION in self.data: return

        # load the file names
        patient_paths = sorted(glob.glob(self.location+'/stage1/*'))
        print len(patient_paths), "patients found"

        # load labels
        self.labels = self.load_labels()

        # make a stratified validation set
        # note, the seed decides the validation set, but it is deterministic in the file_names and labels
        random.seed(317070)
        ids_per_label = [[patient_id for patient_id,label in self.labels.iteritems() if label==l] for l in [0,1]]
        validation_patients = sum([random.sample(sorted(ids), int(VALIDATION_SET_SIZE*len(ids))) for ids in ids_per_label],[])

        # make the static data empty
        for s in self.datasets: self.data[s] = []

        # load metadata
        # with gzip.open(self.location + "metadata.pkl.gz", "rb") as f:
        with open(self.location + "metadata.pkl", "rb") as f:
                self.metadata = cPickle.load(f)

        # print len(spacings)
        # load the filenames and put into the right dataset
        for i, patient_folder in enumerate(patient_paths):
            patient_id = str(patient_folder.split(path.sep)[-1])
            if self.remove_corrupt and patient_id in CORRUPT:
                continue
            if patient_id in self.labels:
                if patient_id in validation_patients:
                    dataset = VALIDATION
                else:
                    dataset = TRAIN
            else:
                dataset = TEST

            self.data[dataset].append(patient_folder)

        # give every patient a unique number
        last_index = -1
        for set in self.datasets:
            self.indices[set] = range(last_index+1,last_index+1+len(self.data[set]))
            if len(self.indices[set]) > 0:
                last_index = self.indices[set][-1]
            print set, len(self.indices[set]), "samples"

    def load_sample(self, sample_id, input_keys_to_do, output_keys_to_do):
        ###################
        #   for testing   #
        ###################
        #sample_id = 1  # test optimizing of parameters
        #import random
        #sample_id = random.choice([1,20000])  # test overfitting

        # find which set this sample is in
        set, set_indices = None, None
        for set, set_indices in self.indices.iteritems():
            if sample_id in set_indices: break
        assert sample_id in set_indices, "Sample ID %d is not known in any of the sets?" % sample_id
        sample_index = set_indices.index(sample_id)

        # prepare empty dicts which will contain the result
        sample = dict()
        sample[INPUT] = dict()
        sample[OUTPUT] = dict()

        patient_path = self.data[set][sample_index]
        patient_name = patient_path.split(path.sep)[-1]

        try:
            data_3d = bcolz.open(patient_path, 'r')
        except:
            print patient_name, "failed to load"
            raise

        meta = self.metadata[patient_name]
        imagepositions = [i[2] for i in meta["imagepositions"]]
        sort_ids = np.argsort(imagepositions)
        intercepts = np.asarray(meta["rescaleintercept"])[sort_ids]
        slopes = np.asarray(meta["rescaleslope"])[sort_ids]

        # Iterate over input tags and return a dict with the requested tags filled
        for tag in input_keys_to_do:
            tags = tag.split(':')
            if "bcolzstage1" not in tags: continue

            if "patient_id" in tags:
                sample[INPUT][tag] = patient_name

            if "3d" in tags or "default" in tags:
                sample[INPUT][tag] = data_3d[:].T[...,sort_ids] # ZYX to XYZ and sort

            if "intercept" in tags:
                sample[INPUT][tag] = intercepts[0]

            if "slope" in tags:
                sample[INPUT][tag] = slopes[0]

            if "pixelspacing" in tags:
                imagepositions.sort()
                sample[INPUT][tag] = meta["pixelspacing"][::-1] + [imagepositions[1] - imagepositions[0]]

            if "shape" in tags:
                sample[INPUT][tag] = data_3d.shape

        for tag in output_keys_to_do:
            tags = tag.split(':')
            if "bcolzstage1" not in tags: continue

            if "target" in tags:
                sample[OUTPUT][tag] = np.int64(self.labels[patient_name])

            if "sample_id" in tags:
                sample[OUTPUT][tag] = sample_id

        return sample

    @staticmethod
    def load_labels():
        labels = dict()
        with open(paths.LABELS_PATH, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip the header
            for row in reader:
                labels[str(row[0])] = int(row[1])
        return labels


def test_loader():
    from application.preprocessors.augmentation_3d import Augment3D
    from application.preprocessors.normalize_scales import DefaultNormalizer
    from application.preprocessors.stage1_to_HU import Stage1ToHU
    nn_input_shape = (256, 256, 80)
    norm_patch_shape = (340, 340, 320)  # median
    preprocessors = [
        Stage1ToHU(tags=["bcolzstage1:3d"]),
        Augment3D(
            tags=["bcolzstage1:3d"],
            output_shape=nn_input_shape,
            norm_patch_shape=norm_patch_shape,
            augmentation_params={
                "scale": [1, 1, 1],  # factor
                "uniform scale": 1,  # factor
                "rotation": [5, 5, 5],  # degrees
                "shear": [0, 0, 0],  # deg
                "translation": [50, 50, 50],  # mm
                "reflection": [0, 0, 0]},  # Bernoulli p
            interp_order=1),
        DefaultNormalizer(tags=["bcolzstage1:3d"])
    ]

    # paths.ALL_DATA_PATH = "/home/lio/data/dsb3/stage1+luna_bcolz/",
    # paths.SPACINGS_PATH =  "/home/lio/data/dsb3/spacings.pkl.gz",
    l = BcolzStage1DataLoader(
        multiprocess=False,
        sets=TRAINING,
        preprocessors=preprocessors)
    l.prepare()

    chunk_size = 1

    batches = l.generate_batch(
        chunk_size=chunk_size,
        required_input={
            # "bcolzstage1:filename": None,
            "bcolzstage1:intercept": (chunk_size,), "bcolzstage1:slope": (chunk_size,), "bcolzstage1:pixelspacing": (chunk_size, 3), "bcolzstage1:3d":(chunk_size,)+nn_input_shape},
        required_output= dict()#{"bcolzstage1:target":None, "bcolzstage1:sample_id":None} # {"luna:segmentation":None, "luna:sample_id":None},
    )

    # import sklearn.metrics
    # lbls = l.labels[VALIDATION]
    # preds = [0.25 for _ in lbls]
    # print sklearn.metrics.log_loss(lbls, preds)

    # sample = l.load_sample(l.indices[TRAIN][0], ["bcolzstage1:3d", "pixelspacing"], ["target"])
    for sample in batches:
        import utils.plt
        print sample[INPUT]["bcolzstage1:3d"].shape, sample[INPUT]["bcolzstage1:pixelspacing"]
        # print l.data[TRAINING][sample[OUTPUT]["bcolzstage1:sample_id"]]
        # print sample[INPUT]["bcolzstage1:filename"], sample[OUTPUT]["bcolzstage1:target"]
        utils.plt.show_animate(np.clip(sample[INPUT]["bcolzstage1:3d"][0]+0.25,0,1), 50, normalize=False)


if __name__ == '__main__':
    test_loader()
