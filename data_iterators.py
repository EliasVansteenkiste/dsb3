import numpy as np
import utils_lung
import pathfinder


class LunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):

        if patient_ids:
            self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.mhd' in p]

        self.id2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
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
                transform_matrices = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    id = utils_lung.luna_extract_pid(patient_path)
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                    x_batch[i, 0, :, :, :], \
                    y_batch[i, 0, :, :, :], \
                    annotations_i, \
                    transformation_i = self.data_prep_fun(data=img,
                                                          pixel_spacing=pixel_spacing,
                                                          luna_annotations=
                                                          self.id2annotations[
                                                              id],
                                                          luna_origin=origin)
                    annotations.append(annotations_i)
                    transform_matrices.append(transformation_i)
                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids, annotations, transform_matrices
                else:
                    yield x_batch, y_batch, patients_ids, annotations, transform_matrices

            if not self.infinite:
                break


class ScanPositiveLunaDataGenerator(LunaDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):
        super(ScanPositiveLunaDataGenerator, self).__init__(data_path, batch_size, transform_params, data_prep_fun, rng,
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


class ValidPatchPositiveLunaDataGenerator(object):
    def __init__(self, data_path, transform_params, patient_ids, data_prep_fun, **kwargs):

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

        self.id2positive_annotations = {}
        self.id2patient_path = {}
        n_positive = 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                n_pos = len(id2positive_annotations[pid])
                self.id2patient_path[pid] = data_path + '/' + pid + '.mhd'
                n_positive += n_pos

        self.nsamples = n_positive
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):

        for pid in self.id2positive_annotations.iterkeys():
            for patch_center in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]
                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)

                patient_annotations = self.id2positive_annotations[pid]
                x_batch, y_batch = self.data_prep_fun(data=img,
                                                      patch_center=patch_center,
                                                      pixel_spacing=pixel_spacing,
                                                      luna_annotations=patient_annotations,
                                                      luna_origin=origin)
                x_batch = np.float32(x_batch)[None, None, :, :, :]
                y_batch = np.float32(y_batch)[None, None, :, :, :]
                yield x_batch, y_batch, [pid]


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


class CandidatesLunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, patient_ids, data_prep_fun, rng,
                 full_batch, random, infinite, positive_proportion, **kwargs):

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.patient_paths = []
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                self.id2negative_annotations[pid] = id2negative_annotations[pid]
                self.patient_paths.append(data_path + '/' + pid + self.file_extension)
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

                    id = utils_lung.luna_extract_pid(patient_path, self.file_extension)
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                        if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
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

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
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

                self.id2patient_path[pid] = data_path + '/' + pid + self.file_extension
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

        for pid in self.id2positive_annotations.iterkeys():
            for patch_center in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path)
                y_batch = np.array([[1.]], dtype='float32')
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]

            for patch_center in self.id2negative_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                y_batch = np.array([[0.]], dtype='float32')
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]


class CandidatesLunaDataGeneratorTODO(object):
    def __init__(self, data_path, transform_params, id2candidates, data_prep_fun, **kwargs):

        self.id2candidates = id2candidates

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

                self.id2patient_path[pid] = data_path + '/' + pid + '.mhd'
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

        for pid in self.id2positive_annotations.iterkeys():
            for patch_center in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                y_batch = np.array([[1.]], dtype='float32')
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]

            for patch_center in self.id2negative_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                y_batch = np.array([[0.]], dtype='float32')
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]
