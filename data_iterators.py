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
            self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.mhd' in p]

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
                    id = utils_lung.luna_extract_pid(patient_path)
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
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
        patient_ids_all = [utils_lung.luna_extract_pid(p) for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids_pos]
        self.nsamples = len(self.patient_paths)


class PatchPositiveLunaDataGenerator(LunaDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):
        super(PatchPositiveLunaDataGenerator, self).__init__(data_path, batch_size, transform_params, data_prep_fun,
                                                             rng,
                                                             full_batch, random, infinite, patient_ids, **kwargs)
        patient_ids_all = [utils_lung.luna_extract_pid(p) for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids_pos]
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
                    id = utils_lung.luna_extract_pid(patient_path)
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)

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


class PatchCentersPositiveLunaDataGenerator(LunaDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):
        super(PatchCentersPositiveLunaDataGenerator, self).__init__(data_path, batch_size, transform_params,
                                                                    data_prep_fun,
                                                                    rng,
                                                                    full_batch, random, infinite, patient_ids, **kwargs)
        patient_ids_all = [utils_lung.luna_extract_pid(p) for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids_pos]
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
                y_batch = np.zeros((nb, 3), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    id = utils_lung.luna_extract_pid(patient_path)
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)

                    patient_annotations = self.id2annotations[id]
                    patch_center = patient_annotations[self.rng.randint(len(patient_annotations))]
                    x_batch[i, 0, :, :, :], zyxd = self.data_prep_fun(data=img,
                                                                      patch_center=patch_center,
                                                                      pixel_spacing=pixel_spacing,
                                                                      luna_annotations=patient_annotations,
                                                                      luna_origin=origin)
                    y_batch[i] = zyxd[:3]


                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break
