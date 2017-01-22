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
    def __init__(self, data_path, batch_size, transform_params, patient_ids=None,
                 full_batch=False, random=True, infinite=True, min_slices=0,
                 data_prep_fun=data_transforms.transform_scan3d, **kwargs):

        if patient_ids:
            self.patient_paths = [p + '.mhd' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.mhd' in p]

        self.id2annotations = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)
        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
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
                x_batch = np.zeros((nb,) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb,) + self.transform_params['patch_size'], dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    id = os.path.basename(patient_path).replace('.mhd', '')
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                    x_batch[i], y_batch[i] = self.data_prep_fun(data=img,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_annotations=self.id2annotations[id],
                                                                luna_origin=origin)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_batch], [y_batch], patients_ids
                else:
                    yield [x_batch], [y_batch], patients_ids

            if not self.infinite:
                break
