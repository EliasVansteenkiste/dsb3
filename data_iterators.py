import data_transforms
import glob
import re
import itertools
from collections import defaultdict
import numpy as np
import utils
import utils_lung
import pathfinder
import os


class LunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):


        if patient_ids:
            self.patient_paths = [data_path + '/' + p + '.pkl' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.pkl' in p]

        self.id2annotations = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)
        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.batch_size = batch_size
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        print 'generate bitch'
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                patients_ids = []
                annotations = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    id = utils_lung.luna_extract_pid(patient_path,'.pkl')

                    patients_ids.append(id)
                    print 'patiend id', id 
                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path)
                    print 'reading done'
                    x_batch[i, 0, :, :, :], y_batch[i, 0, :, :, :], annotations_i = self.data_prep_fun(data=img,
                                                                                                       pixel_spacing=pixel_spacing,
                                                                                                       luna_annotations=
                                                                                                       self.id2annotations[
                                                                                                           id],
                                                                                                       luna_origin=origin)
                    annotations.append(annotations_i)
                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids, annotations
                else:
                    yield x_batch, y_batch, patients_ids, annotations

            if not self.infinite:
                break


class PositiveLunaDataGenerator(LunaDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):
        super(PositiveLunaDataGenerator, self).__init__(data_path, batch_size, transform_params, data_prep_fun, rng,
                                                        full_batch, random, infinite, patient_ids, **kwargs)
        patient_ids_all = [utils_lung.luna_extract_pid(p,'.pkl') for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.pkl' for p in patient_ids_pos]
        self.nsamples = len(self.patient_paths)


class PatchPositiveLunaDataGenerator(LunaDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):
        super(PatchPositiveLunaDataGenerator, self).__init__(data_path, batch_size, transform_params, data_prep_fun,
                                                             rng,
                                                             full_batch, random, infinite, patient_ids, **kwargs)
        patient_ids_all = [utils_lung.luna_extract_pid(p,'.pkl') for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.pkl' for p in patient_ids_pos]
        self.nsamples = len(self.patient_paths)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    id = utils_lung.luna_extract_pid(patient_path,'.pkl')
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path)

                    patient_annotations = self.id2annotations[id]
                    patch_center = patient_annotations[self.rng.randint(len(patient_annotations))]
                    x_batch[i, 0, :, :, :], y_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                                        patch_center=patch_center,
                                                                                        pixel_spacing=pixel_spacing,
                                                                                        luna_annotations=patient_annotations,
                                                                                        luna_origin=origin)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break

class Luna_DG_Elias(LunaDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, positive_proportion=0.5, **kwargs):
        super(Luna_DG_Elias, self).__init__(data_path, batch_size, transform_params, data_prep_fun,
                                                             rng, 
                                                             full_batch, random, infinite, patient_ids, **kwargs)
        patient_ids_all = [utils_lung.luna_extract_pid(p,'.pkl') for p in self.patient_paths]

        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.pkl' for p in patient_ids_pos]
        self.nsamples = len(self.patient_paths)
        self.id2_no_nodules, self.id2_nodules  = utils_lung.read_luna_candidates('candidates_V2.csv')
        self.positive_proportion = positive_proportion

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb,1,) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb,1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    id = utils_lung.luna_extract_pid(patient_path,'.pkl')
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path)

                    patient_nodules = self.id2_nodules[id]
                    patient_no_nodules = self.id2_no_nodules[id]
                    candidate = None
                    if len(patient_nodules)>0 and np.random.choice([0, 1], p=[1-self.positive_proportion, self.positive_proportion]):
                        candidate = patient_nodules[self.rng.randint(len(patient_nodules))]
                        y_batch[i] = 1.

                    else:
                        candidate = patient_no_nodules[self.rng.randint(len(patient_no_nodules))]
                        y_batch[i] = 0.

                    x_batch[i, :, :, :] = self.data_prep_fun(data=img, patch_center=candidate,
                                                                          pixel_spacing=pixel_spacing,
                                                                          luna_origin=origin)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break



class CandidatesLunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, patient_ids, data_prep_fun, rng,
                 full_batch, random, infinite, positive_proportion, **kwargs):

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.patient_paths = []
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                self.id2negative_annotations[pid] = id2negative_annotations[pid]
                self.patient_paths.append(data_path + '/' + pid + '.pkl')
                n_positive += len(id2positive_annotations[pid])
                n_negative += len(id2negative_annotations[pid])

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.patient_paths)

        print 'n patients', self.nsamples
        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.batch_size = batch_size
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.positive_proportion = positive_proportion

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    id = utils_lung.luna_extract_pid(patient_path,'.pkl')
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path)
                    if i < np.rint(self.batch_size * self.positive_proportion):
                        patient_annotations = self.id2positive_annotations[id]
                    else:
                        patient_annotations = self.id2negative_annotations[id]

                    patch_center = patient_annotations[self.rng.randint(len(patient_annotations))]

                    y_batch[i] = float(patch_center[-1] > 0)
                    x_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                patch_center=patch_center,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_origin=origin)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break


class CandidatesLunaValidDataGenerator(object):
    def __init__(self, data_path, transform_params, patient_ids, data_prep_fun, **kwargs):
        rng = np.random.RandomState(42)  # do not change this!!!

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.id2patient_path = {}
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                negative_annotations = id2negative_annotations[pid]
                n_pos = len(id2positive_annotations[pid])
                n_neg = len(id2negative_annotations[pid])
                neg_idxs = rng.choice(n_neg, size=n_pos, replace=False)
                negative_annotations_selected = []
                for i in neg_idxs:
                    negative_annotations_selected.append(negative_annotations[i])
                self.id2negative_annotations[pid] = negative_annotations_selected

                self.id2patient_path[pid] = data_path + '/' + pid + '.pkl'
                n_positive += n_pos
                n_negative += n_pos

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.rng = rng
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        x_batch = np.zeros((1, 1) + self.transform_params['patch_size'], dtype='float32')
        y_batch = np.zeros((1, 1), dtype='float32')

        for pid in self.id2positive_annotations.iterkeys():
            for patch_center in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path)
                y_batch[0] = 1.
                x_batch[0, 0, :, :, :] = self.data_prep_fun(data=img,
                                                            patch_center=patch_center,
                                                            pixel_spacing=pixel_spacing,
                                                            luna_origin=origin)

                yield x_batch, y_batch, [pid]

            for patch_center in self.id2negative_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path)
                y_batch[0] = 0.
                x_batch[0, 0, :, :, :] = self.data_prep_fun(data=img,
                                                            patch_center=patch_center,
                                                            pixel_spacing=pixel_spacing,
                                                            luna_origin=origin)

                yield x_batch, y_batch, [pid]