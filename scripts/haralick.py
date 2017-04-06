import numpy as np

import sys
sys.path.append(".")
import data_transforms
import data_iterators
import pathfinder
from collections import namedtuple
from functools import partial
import utils
import utils_lung
import os
import buffering

import mahotas as mh


p_transform = {'patch_size': (64, 64, 64),
               'mm_patch_size': (48, 48, 48),
               'pixel_spacing': (1., 1., 1.)
               }

n_candidates_per_patient = 8
rng = np.random.RandomState(42)


def data_prep_function(data, patch_centers, pixel_spacing, p_transform,
                       p_transform_augment, **kwargs):
    x = data_transforms.transform_dsb_candidates(data=data,
                                                 patch_centers=patch_centers,
                                                 p_transform=p_transform,
                                                 p_transform_augment=p_transform_augment,
                                                 pixel_spacing=pixel_spacing,
                                                 order=0,
                                                 concat=False)
    # x = data_transforms.pixelnormHU(x)
    return x


data_prep_function = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
candidates_config = 'dsb_c3_s5_p8a1'
candidates_path = predictions_dir + '/%s' % candidates_config
id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

train_valid_ids = utils.load_pkl(pathfinder.MIXED_SPLIT_PATH)
pids = train_valid_ids['training']+train_valid_ids['validation']+train_valid_ids['test']


l = data_iterators.DSBPatientsDataGenerator(data_path=pathfinder.DATA_PATH,
                                                              batch_size=1,
                                                              transform_params=p_transform,
                                                              n_candidates_per_patient=n_candidates_per_patient,
                                                              data_prep_fun=data_prep_function,
                                                              id2candidates_path=id2candidates_path,
                                                              rng=rng,
                                                              patient_ids=pids,
                                                              random=False, infinite=True)
patient_paths = l.patient_paths
nsamples = l.nsamples
features = {}
from time import time, sleep
from signal import signal, SIGKILL, SIGINT
import multiprocessing as mp

def extract_haralick(idx):
    patient_path = patient_paths[idx]
    pid = utils_lung.extract_pid_dir(patient_path)


    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

    all_candidates = utils.load_pkl(l.id2candidates_path[pid])
    top_candidates = all_candidates[:l.n_candidates_per_patient]

    candidate_patches = l.data_prep_fun(data=img,
                                        patch_centers=top_candidates,
                                        pixel_spacing=pixel_spacing)
    feats = []
    # t0 = time()
    for patch in candidate_patches:
        patch = np.clip(patch.astype("int32"), -1000, 400)
        patch += 1000
        # print patch.dtype, patch.shape, patch.min(), patch.max()
        haralick = mh.features.haralick(patch, ignore_zeros=False,
                                        preserve_haralick_bug=False, compute_14th_feature=False).flatten()

        feats.append(haralick.astype("float32"))
    # print "  ", time() - t0, "seconds"
    return pid, feats


def terminate(sig=None, frame=None):
    os.killpg(0, sig)
    sleep(1)  # give some time until we escalate to kill -9
    os.killpg(0, SIGKILL)

signal(SIGINT, terminate)

pool = mp.Pool(4)
res = [pool.apply_async(extract_haralick, (i,)) for i in range(nsamples)]
t0 = time()
for i, r in enumerate(res):
    pid, feats = r.get()
    print i + 1, "/", nsamples, pid, len(feats), (time() - t0)/(i+1), "s/patient"
    features[pid] = feats

# import cPickle
# with open("haralickv1.pkl", "wb") as f:
#     cPickle.dump(features, f, protocol=cPickle.HIGHEST_PROTOCOL)


import cPickle
# import pathfinder
from sklearn import preprocessing as prep

# with open("haralickv1.pkl", "rb") as f:
#     features = cPickle.load(f)

nodule_feats = np.array([nodule for feats in features.values() for nodule in feats], "float32")
print len(nodule_feats)

scaler = prep.StandardScaler().fit(nodule_feats)

for pid, feats in features.items():
    x = np.array(feats, dtype="float32")
    x = scaler.transform(x)
    features[pid] = x

with open(pathfinder.HARALICK_PATH, "wb") as f:
    cPickle.dump(features, f, protocol=cPickle.HIGHEST_PROTOCOL)


