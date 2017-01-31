import glob
import csv
import random
from os import path
import os
import bcolz
import sys
import cPickle
import gzip

import numpy as np

from interfaces.data_loader import StandardDataLoader, TRAINING, VALIDATION, TEST, INPUT, OUTPUT, TRAIN
from utils import paths


VALIDATION_SET_SIZE = 0.2


class BcolzAllDataLoader(StandardDataLoader):

    # These are shared between all objects of this type
    labels = dict()
    names = dict()
    spacings = dict()

    datasets = [TRAIN, VALIDATION, TEST]

    def __init__(self,
                 use_luna=False,
                 location=paths.ALL_DATA_PATH, *args, **kwargs):
        super(BcolzAllDataLoader, self).__init__(location=location, *args, **kwargs)
        self.use_luna = use_luna

    def prepare(self):
        """
        Prepare the dataloader, by storing values to static fields of this class
        In this case, only filenames are loaded prematurely
        :return:
        """
        
        print "previous bcolz nthreads:", bcolz.set_nthreads(1);

        # step 0: load only when not loaded yet
        if TRAINING in self.data and VALIDATION in self.data: return

        # step 1: load the file names
        patients = sorted(glob.glob(self.location+'/*/'))

        print len(patients), "patients"
        # sys.exit()


        labels = dict()
        with open(paths.LABELS_PATH, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip the header
            for row in reader:
                labels[str(row[0])] = int(row[1])

        # make a stratified validation set
        # note, the seed decides the validation set, but it is deterministic in the file_names and labels
        random.seed(317070)
        ids_per_label = [[patient_id for patient_id,label in labels.iteritems() if label==l] for l in [0,1]]
        validation_patients = sum([random.sample(sorted(ids), int(VALIDATION_SET_SIZE*len(ids))) for ids in ids_per_label],[])

        if self.use_luna:
            luna_labels = load_luna_labels(patients)
            print len(luna_labels), "luna labels added"
            labels.update(luna_labels)

        # make the static data empty
        for s in self.datasets:
            self.data[s] = []
            self.labels[s] = []
            self.names[s] = []
            self.spacings[s] = []

        with gzip.open(paths.SPACINGS_PATH) as f:
            spacings = cPickle.load(f)

        # load the filenames and put into the right dataset
        for i, patient_folder in enumerate(patients):
            patient_id = str(patient_folder.split(path.sep)[-2])
            if patient_id in labels:
                if patient_id in validation_patients:
                    dataset = VALIDATION
                else:
                    dataset = TRAIN
            else:
                dataset = TEST


            self.data[dataset].append(patient_folder)
            if patient_id in labels:
                self.labels[dataset].append(labels[patient_id])
            self.names[dataset].append(patient_id)
            self.spacings[dataset].append(spacings[patient_id])

        print "train", len(self.data[TRAIN])
        print "valid", len(self.data[VALIDATION])
        print "test", len(self.data[TEST])

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
            if sample_id in set_indices:
                break

        assert sample_id in set_indices, "Sample ID %d is not known in any of the sets?" % sample_id

        sample_index = set_indices.index(sample_id)

        # prepare empty dicts which will contain the result
        sample = dict()
        sample[INPUT] = dict()
        sample[OUTPUT] = dict()

        patient_name = self.names[set][sample_index]
        try:
            carray = bcolz.open(self.data[set][sample_index], 'r')
            volume = carray[:].T  # move from zyx to xyz
            carray.free_cachemem()
            del carray
        except:
            print patient_name
            raise

        # Iterate over input tags and return a dict with the requested tags filled
        for tag in input_keys_to_do:
            tags = tag.split(':')
            if "bcolzall" not in tags: continue

            if "filename" in tags:
                sample[INPUT][tag] = patient_name

            if "3d" in tags or "default" in tags:
                sample[INPUT][tag] = volume

            if "pixelspacing" in tags:
                sample[INPUT][tag] = self.spacings[set][sample_index][::-1]  # in mm per pixel

            if "shape" in tags:
                sample[INPUT][tag] = volume.shape

        for tag in output_keys_to_do:
            tags = tag.split(':')
            if "bcolzall" not in tags: continue

            if "target" in tags:
                sample[OUTPUT][tag] = np.int64(self.labels[set][sample_index])

            if "sample_id" in tags:
                sample[OUTPUT][tag] = sample_id

        return sample


def load_luna_labels(patients):
    luna_labels = {}
    patient_ids = set([str(p.split(path.sep)[-2]) for p in patients])
    with open(paths.LUNA_LABELS_PATH, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)  # skip the header
        for row in reader:
            name = str(row[0])
            if name not in patient_ids: continue
            if name not in luna_labels:
                luna_labels[name] = [diameter_to_prob(float(row[4]))]
            else:
                luna_labels[name].append(diameter_to_prob(float(row[4])))
    for name, probs in luna_labels.items():
        probs = np.asarray(probs)
        prob = 1. - np.prod(1. - probs)  # nodules assumed independent
        luna_labels[name] = prob



    return luna_labels


# 6% to 28% for nodules 5 to 10 mm,
prob5 = (0.01+0.06)/2.
slope10 = (0.28-prob5) / (10.-5.)
offset10 = prob5 - slope10*5.

slope20 = (0.64-0.28) / (20.-10.)
offset20 = 0.28 - slope20*10.

# and 64% to 82% for nodules >20 mm in diameter
slope25 = (0.82-0.64) / (25.-20.)
offset25 = 0.64 - slope25*20.

slope30 = (0.93-0.82) / (30.-25.)
offset30 = 0.82 - slope30*25.

# For nodules more than 3 cm in diameter, 93% to 97% are malignant
slope40 = (0.97-0.93) / (40.-30.)
offset40 = 0.93 - slope40*30.

def diameter_to_prob(diam):
    # The prevalence of malignancy is 0% to 1% for nodules <5 mm,
    if diam < 5:
        p = prob5*diam/5.
    elif diam < 10:
        p = slope10*diam+offset10
    elif diam < 20:
        p = slope20*diam+offset20
    elif diam < 25:
        p = slope25*diam+offset25
    elif diam < 30:
        p = slope30*diam+offset30
    else:
        p = slope40 * diam + offset40
    return np.clip(p ,0.,1.)


def test_diameter_to_prob():
    n = 1000
    xs = [i/float(n)*40. for i in range(n)]
    pnts = [diameter_to_prob(x) for x in xs]
    import matplotlib.pyplot as plt
    plt.xlabel("diameter (mm)")
    plt.ylabel("cancer prob")
    plt.plot(xs, pnts)
    plt.show()

    patients = sorted(glob.glob(paths.ALL_DATA_PATH + '/*/'))
    lbls = load_luna_labels(patients)

    for name, prob in lbls.items(): print name, prob


def test_diagnosis():
    patients = sorted(glob.glob(paths.ALL_DATA_PATH + '/*/'))
    patient_ids = set([str(p.split(path.sep)[-2]) for p in patients])
    print len(patient_ids)
    lbls = {}
    with open(paths.DIAGNOSIS_PATH, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip the header
        for row in reader:
            name = str(row[0])
            if name not in patient_ids:
                print name
            else:
                lbls[name] = int(row[1])
    print len(lbls), lbls


def test_loader():
    from application.preprocessors.augmentation_3d import Augment3D
    from application.preprocessors.normalize_scales import DefaultNormalizer
    nn_input_shape = (128, 128, 64)
    norm_patch_shape = (340, 340, 320)  # median
    preprocessors = [
        Augment3D(
            tags=["bcolzall:3d"],
            output_shape=nn_input_shape,
            norm_patch_shape=norm_patch_shape,
            augmentation_params={
                "scale": [1.05, 1.05, 1.05],  # factor
                "uniform scale": 1.2,  # factor
                "rotation": [5, 5, 5],  # degrees
                "shear": [3, 3, 3],  # deg
                "translation": [50, 50, 50],  # mm
                "reflection": [0, 0, 0]},  # Bernoulli p
            interp_order=1),
        DefaultNormalizer(tags=["bcolzall:3d"])
    ]

    # paths.ALL_DATA_PATH = "/home/lio/data/dsb3/stage1+luna_bcolz/",
    # paths.SPACINGS_PATH =  "/home/lio/data/dsb3/spacings.pkl.gz",
    l = BcolzAllDataLoader(
        multiprocess=False,
        location="/mnt/storage/data/dsb3/stage1+luna_bcolz/",
        sets=TRAINING,
        preprocessors=preprocessors)
    l.prepare()

    chunk_size = 1

    batches = l.generate_batch(
        chunk_size=chunk_size,
        required_input={"bcolzall:pixelspacing": (chunk_size, 3), "bcolzall:3d":(chunk_size,)+nn_input_shape},
        required_output=dict()  # {"luna:segmentation":None, "luna:sample_id":None},
    )

    # import sklearn.metrics
    # lbls = l.labels[VALIDATION]
    # preds = [0.25 for _ in lbls]
    # print sklearn.metrics.log_loss(lbls, preds)

    # sample = l.load_sample(l.indices[TRAIN][0], ["bcolzall:3d", "pixelspacing"], ["target"])
    for sample in batches:
        import utils.plt
        print sample[INPUT]["bcolzall:3d"].shape, sample[INPUT]["bcolzall:pixelspacing"]
        utils.plt.show_animate(sample[INPUT]["bcolzall:3d"][0], 50)


if __name__ == '__main__':
    test_loader()
    # test_diameter_to_prob()
    # test_diagnosis()
