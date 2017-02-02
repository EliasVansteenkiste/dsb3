import numpy as np
from os import path
import csv
import glob
import string

from interfaces.preprocess import BasePreprocessor
from interfaces.data_loader import INPUT, OUTPUT
from utils import paths


# Hounsfield Unit
class LunaDiagnose(BasePreprocessor):
    def __init__(self, tags):
        self.tags = tags

        patients = sorted(glob.glob(paths.BCOLZ_DATA_PATH + '/luna/*/'))
        cancer_prob = {}
        patient_ids = set([str(p.split(path.sep)[-1]) for p in patients])

        with open(paths.LUNA_LABELS_PATH, "rb") as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip the header
            for row in reader:
                name = str(row[0])
                if name not in patient_ids: continue
                if name not in cancer_prob:
                    cancer_prob[name] = [diameter_to_prob(float(row[4]))]
                else:
                    cancer_prob[name].append(diameter_to_prob(float(row[4])))

        for name, probs in cancer_prob.items():
            probs = np.asarray(probs)
            prob = 1. - np.prod(1. - probs)  # nodules assumed independent
            cancer_prob[name] = prob

        self.cancer_prob = cancer_prob

    @property
    def extra_output_tags_required(self):
        datasetnames = set()
        for tag in self.tags: datasetnames.add(tag.split(':')[0])
        output_tags_extra = [dsn + ":target" for dsn in datasetnames]
        return output_tags_extra

    @property
    def extra_input_tags_required(self):
        datasetnames = set()
        for tag in self.tags: datasetnames.add(tag.split(':')[0])
        input_tags_extra = [dsn + ":patient_id" for dsn in datasetnames]
        return input_tags_extra

    def process(self, sample):
        for tag in self.tags:
            targettag = tag.split(':')[0] + ":target"
            assert targettag in sample[OUTPUT], "tag %s not found"%targettag
            patientidtag = tag.split(':')[0] + ":patient_id"
            assert patientidtag in sample[INPUT], "tag %s not found" % patientidtag

            patientid = sample[INPUT][patientidtag]
            if not "." in patientid: continue # not Luna

            if patientid in self.cancer_prob:
                sample[OUTPUT][targettag] = self.cancer_prob[patientid]
            else:
                sample[OUTPUT][targettag] = 0.25


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