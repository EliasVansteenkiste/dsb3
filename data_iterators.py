import numpy as np
import utils_lung
import pathfinder
import utils


class LunaDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, rng,
                 random, infinite, patient_ids=None, **kwargs):

        if patient_ids:
            self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.mhd' in p]

        self.id2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs)):
                idx = rand_idxs[pos]

                patient_path = self.patient_paths[idx]
                pid = utils_lung.extract_pid_filename(patient_path)

                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                x, y, annotations, tf_matrix = self.data_prep_fun(data=img,
                                                                  pixel_spacing=pixel_spacing,
                                                                  luna_annotations=
                                                                  self.id2annotations[pid],
                                                                  luna_origin=origin)

                x = np.float32(x)[None, None, :, :, :]
                y = np.float32(y)[None, None, :, :, :]

                yield x, y, None, annotations, tf_matrix, pid

            if not self.infinite:
                break


class LunaScanPositiveDataGenerator(LunaDataGenerator):
    def __init__(self, data_path, transform_params, data_prep_fun, rng,
                 random, infinite, patient_ids=None, **kwargs):
        super(LunaScanPositiveDataGenerator, self).__init__(data_path, transform_params, data_prep_fun, rng,
                                                            random, infinite, patient_ids, **kwargs)
        patient_ids_all = [utils_lung.extract_pid_filename(p) for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids_pos]
        self.nsamples = len(self.patient_paths)


class LunaScanPositiveLungMaskDataGenerator(LunaScanPositiveDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):
        super(LunaScanPositiveLungMaskDataGenerator, self).__init__(data_path, transform_params,
                                                                    data_prep_fun, rng,
                                                                    random, infinite, patient_ids, **kwargs)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs)):
                idx = rand_idxs[pos]

                patient_path = self.patient_paths[idx]
                pid = utils_lung.extract_pid_filename(patient_path)

                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                x, y, lung_mask, annotations, tf_matrix = self.data_prep_fun(data=img,
                                                                             pixel_spacing=pixel_spacing,
                                                                             luna_annotations=
                                                                             self.id2annotations[pid],
                                                                             luna_origin=origin)

                x = np.float32(x)[None, None, :, :, :]
                y = np.float32(y)[None, None, :, :, :]
                lung_mask = np.float32(lung_mask)[None, None, :, :, :]

                yield x, y, lung_mask, annotations, tf_matrix, pid

            if not self.infinite:
                break


class PatchPositiveLunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):

        self.id2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

        if patient_ids:
            self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.mhd' in p]

        patient_ids_all = [utils_lung.extract_pid_filename(p) for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids_pos]

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.batch_size = batch_size
        self.full_batch = full_batch

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
                    id = utils_lung.extract_pid_filename(patient_path)
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

                    id = utils_lung.extract_pid_filename(patient_path, self.file_extension)
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

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
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


class FixedCandidatesLunaDataGenerator(object):
    def __init__(self, data_path, transform_params, id2candidates_path, data_prep_fun, top_n=None):

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2candidates_path = id2candidates_path
        self.id2patient_path = {}
        for pid in id2candidates_path.keys():
            self.id2patient_path[pid] = data_path + '/' + pid + self.file_extension

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.top_n = top_n

    def generate(self):

        for pid in self.id2candidates_path.iterkeys():
            patient_path = self.id2patient_path[pid]
            print 'PATIENT', pid
            candidates = utils.load_pkl(self.id2candidates_path[pid])
            if self.top_n is not None:
                candidates = candidates[:self.top_n]
                print candidates
            print 'n blobs', len(candidates)

            img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)

            for candidate in candidates:
                y_batch = np.array(candidate, dtype='float32')
                patch_center = candidate[:3]
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]


class DSBScanDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, **kwargs):
        self.patient_paths = utils_lung.get_patient_data_paths(data_path)
        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        for p in self.patient_paths:
            pid = utils_lung.extract_pid_dir(p)

            img, pixel_spacing = utils_lung.read_dicom_scan(p)

            x, tf_matrix = self.data_prep_fun(data=img, pixel_spacing=pixel_spacing)

            x = np.float32(x)[None, None, :, :, :]
            yield x, None, tf_matrix, pid


class DSBScanLungMaskDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, exclude_pids=None,
                 include_pids=None, part_out_of=(1, 1)):

        self.patient_paths = utils_lung.get_patient_data_paths(data_path)

        this_part = part_out_of[0]
        all_parts = part_out_of[1]
        part_lenght = int(len(self.patient_paths) / all_parts)

        if this_part == all_parts:
            self.patient_paths = self.patient_paths[part_lenght * (this_part - 1):]
        else:
            self.patient_paths = self.patient_paths[part_lenght * (this_part - 1): part_lenght * this_part]

        if exclude_pids is not None:
            for ep in exclude_pids:
                for i in xrange(len(self.patient_paths)):
                    if ep in self.patient_paths[i]:
                        self.patient_paths.pop(i)
                        break

        if include_pids is not None:
            self.patient_paths = [data_path + '/' + p for p in include_pids]

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        for p in self.patient_paths:
            pid = utils_lung.extract_pid_dir(p)

            img, pixel_spacing = utils_lung.read_dicom_scan(p)

            x, lung_mask, tf_matrix = self.data_prep_fun(data=img, pixel_spacing=pixel_spacing)

            x = np.float32(x)[None, None, :, :, :]
            lung_mask = np.float32(lung_mask)[None, None, :, :, :]
            yield x, lung_mask, tf_matrix, pid


            
            
            
class AAPMScanLungMaskDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, exclude_pids=None,
                 include_pids=None, part_out_of=(1, 1)):

        # just hand back the first series here?
        self.patient_paths = utils_lung.get_patient_data_paths_aapm(data_path)

        this_part = part_out_of[0]
        all_parts = part_out_of[1]
        part_lenght = int(len(self.patient_paths) / all_parts)

        if this_part == all_parts:
            self.patient_paths = self.patient_paths[part_lenght * (this_part - 1):]
        else:
            self.patient_paths = self.patient_paths[part_lenght * (this_part - 1): part_lenght * this_part]

        # TODO: ignored this for now
        # if exclude_pids is not None:
        #     for ep in exclude_pids:
        #         for i in xrange(len(self.patient_paths)):
        #             if ep in self.patient_paths[i]:
        #                 self.patient_paths.pop(i)
        #                 break

        # if include_pids is not None:
        #     self.patient_paths = [data_path + '/' + p for p in include_pids]

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        for p in self.patient_paths:
            pid = utils_lung.extract_pid_dir_aapm(p)

            img, pixel_spacing = utils_lung.read_dicom_scan(p)

            x, lung_mask, tf_matrix = self.data_prep_fun(data=img, pixel_spacing=pixel_spacing)

            x = np.float32(x)[None, None, :, :, :]
            lung_mask = np.float32(lung_mask)[None, None, :, :, :]
            yield x, lung_mask, tf_matrix, pid



class CandidatesAAPMDataGenerator(object):
    def __init__(self, data_path, transform_params, id2candidates_path, data_prep_fun, exclude_pids=None):
        if exclude_pids is not None:
            for p in exclude_pids:
                id2candidates_path.pop(p, None)

        self.id2candidates_path = id2candidates_path
        self.id2patient_path = {}
        for pid in id2candidates_path.keys():
            self.id2patient_path[pid] =utils_lung.get_path_to_image_from_patient(data_path,pid)#: data_path + '/' + pid +

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):

        for pid in self.id2candidates_path.iterkeys():
            patient_path = self.id2patient_path[pid]
            print pid, patient_path
            img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

            print self.id2candidates_path[pid]
            candidates = utils.load_pkl(self.id2candidates_path[pid])
            print candidates.shape

            #y_batch=read_aapm_annotations(file_path)

            for candidate in candidates:
                y_batch = np.array(candidate, dtype='float32')
                patch_center = candidate[:3]
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]

            



            
class CandidatesDSBDataGenerator(object):
    def __init__(self, data_path, transform_params, id2candidates_path, data_prep_fun, exclude_pids=None):
        if exclude_pids is not None:
            for p in exclude_pids:
                id2candidates_path.pop(p, None)

        self.id2candidates_path = id2candidates_path
        self.id2patient_path = {}
        for pid in id2candidates_path.keys():
            self.id2patient_path[pid] = data_path + '/' + pid

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):

        for pid in self.id2candidates_path.iterkeys():
            patient_path = self.id2patient_path[pid]
            print pid, patient_path
            img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

            print self.id2candidates_path[pid]
            candidates = utils.load_pkl(self.id2candidates_path[pid])
            print candidates.shape
            for candidate in candidates:
                y_batch = np.array(candidate, dtype='float32')
                patch_center = candidate[:3]
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]


class AAPMPatientsDataGenerator(object):

    def __init__(self, data_path, batch_size, transform_params, id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None):


        self.id2label = utils_lung.read_aapm_labels_per_patient(pathfinder.AAPM_LABELS_PATH)
        self.id2candidates_path = id2candidates_path
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                self.patient_paths.append(utils_lung.get_path_to_image_from_patient(data_path,pid))
        else:
            raise ValueError('provide patient ids')

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite
        
        #TODO: used for debugging purposes to take a look at the ground truth
        self.ground_truth=utils_lung.read_aapm_annotations(pathfinder.AAPM_LABELS_PATH)

        self.shuffle_top_n = shuffle_top_n


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch = np.zeros((self.batch_size, self.n_candidates_per_patient, 1,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    pid = utils_lung.extract_pid_dir_aapm(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    # first take a look at the image 
                    # get the voxel coords of the node to 
                    #print "ground_truth: {}".format(ground_truth)
                    #utils_plots.plot_slice_3d_3axis(img, pid, img_dir=utils.get_dir_path('analysis',pathfinder.METADATA_PATH), idx=ground_truth)



                    # can I probably just use that function to hand down the one center and set world coordinates to false? that should work I think :-)

                    import utils_plots
                    ground_truth=[idx for idx in self.ground_truth[pid][0][:-1]]
                    ground_truth[0]=img.shape[0]-ground_truth[0]
                    

                    print "ground truth: {}".format(ground_truth)
                    # TODO: candidates probably need to be converted to voxel coordinates to 
                    # be comparable ?
                    print "candidates: {}".format(top_candidates)

                    # x_plot = np.float32(self.data_prep_fun(data=img,
                    #                                            patch_centers=[ground_truth],
                    #                                            pixel_spacing=pixel_spacing))[:, None, :, :, :]

                    x_batch[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, None, :, :, :]
                    #utils_plots.plot_slice_3d_3axis(x_batch[i][0,0,:,:,:], pid, img_dir=utils.get_dir_path('analysis',pathfinder.METADATA_PATH))
                    
                    y_batch[i] = self.id2label.get(pid)
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break





class DSBPatientsDataGenerator(object):

    def __init__(self, data_path, batch_size, transform_params, id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None):

        self.id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)
        self.id2candidates_path = id2candidates_path
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                self.patient_paths.append(data_path + '/' + pid)
        else:
            raise ValueError('provide patient ids')

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite

        self.shuffle_top_n = shuffle_top_n


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch = np.zeros((self.batch_size, self.n_candidates_per_patient, 1,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    pid = utils_lung.extract_pid_dir(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    x_batch[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, None, :, :, :]
                    y_batch[i] = self.id2label.get(pid)
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break


class MixedPatientsDataGenerator(object):

    def __init__(self, data_path,aapm_data_path, batch_size, transform_params, id2candidates_path,aapm_id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None,aapm_patient_ids=None):


        # ok we will not have colliding ids

        # build id2label and id2candidate paths
        dsb_id2label=utils_lung.read_labels(pathfinder.LABELS_PATH) 
        aapm_id2label=utils_lung.read_aapm_labels_per_patient(pathfinder.AAPM_LABELS_PATH)
        

        self.id2label = dsb_id2label.copy()#{**dsb_id2label, **aapm_id2label}
        self.id2label.update(aapm_id2label)
        #print "id2label: {}".format(self.id2label)
        self.id2candidates_path = id2candidates_path.copy()  #{id2candidates_path,aapm_id2candidates_path}
        self.id2candidates_path.update(aapm_id2candidates_path)
        #print "id2candidates_path: {}".format(self.id2candidates_path)
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                self.patient_paths.append((pid,data_path + '/' + pid))
        else:
            raise ValueError('provide patient ids')

        if aapm_patient_ids is not None:
           for pid in aapm_patient_ids:
               self.patient_paths.append((pid,utils_lung.get_path_to_image_from_patient(aapm_data_path,pid)))




        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite

        self.shuffle_top_n = shuffle_top_n


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch = np.zeros((self.batch_size, self.n_candidates_per_patient, 1,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    (pid,patient_path) = self.patient_paths[idx]
                    #pid = utils_lung.extract_pid_dir(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    x_batch[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, None, :, :, :]
                    y_batch[i] = self.id2label.get(pid)
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break

# data_prep_fun

class CandidatesMixedMalignantBenignGenerator(object):
    def __init__(self, data_path,aapm_data_path, batch_size, transform_params, patient_ids, aapm_patient_ids, data_prep_fun,
                 rng, full_batch, random, infinite, positive_proportion):
        
        # just the labels
        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2positive_annotations_aapm = utils_lung.read_aapm_annotations(pathfinder.AAPM_LABELS_PATH)
        
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        id2negative_annotations_aapm = utils_lung.read_luna_negative_candidates(pathfinder.AAPM_CANDIDATES_PATH)
#        print "id2negative_annotations_aapm {}".format(id2negative_annotations_aapm)


        self.luna_file_extension = '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.patient_paths = []
        n_positive, n_negative = 0, 0

        # just build the labels beforehand, up here, per pid?


        for pid in patient_ids:

            if pid in id2positive_annotations:

                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                self.id2negative_annotations[pid] = id2negative_annotations[pid]
                self.patient_paths.append((pid,data_path + '/' + pid + self.luna_file_extension))
                n_positive += len(id2positive_annotations[pid])
                n_negative += len(id2negative_annotations[pid])


        for pid in aapm_patient_ids:

            if pid in id2positive_annotations_aapm:

                self.id2positive_annotations[pid] = id2positive_annotations_aapm[pid]

                self.id2negative_annotations[pid] = id2negative_annotations_aapm[pid]
                self.patient_paths.append((pid,utils_lung.get_path_to_image_from_patient(aapm_data_path,pid)))
                n_positive += len(id2positive_annotations[pid])
                n_negative += len(id2negative_annotations[pid])

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.patient_paths)

        print 'n patients', self.nsamples
        self.aapm_patient_ids=aapm_patient_ids
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
                y_batch = np.zeros((nb, 3), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    (id,patient_path) = self.patient_paths[idx]

                    
                    patients_ids.append(id)


                    
                    #print "patient_path: {}".format(patient_path)
                    
                    if '.mhd' in patient_path:
                        img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                        world_coord_system=True
                    else:
                        img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)
                        origin=None
                        world_coord_system=False
                     

                    #y_batch[i] = np.zeros((1,3))



                    if i < np.rint(self.batch_size * self.positive_proportion):
                        patient_annotations = self.id2positive_annotations[id]
                        y_batch[i,0]=1
                    else:
                        patient_annotations = self.id2negative_annotations[id]
                        y_batch[i,0]=0


                    # print "negative annotations: {}".format(self.id2negative_annotations[id])                    
                    # print "patient_annotations: {}".format(patient_annotations)
                    # print "id: {}".format(id)

                    # print "len: {}".format(len(patient_annotations))
                    # print "y_batch: {}".format(y_batch[i,0])

                    random_index=self.rng.randint(len(patient_annotations))
                    patch_center = patient_annotations[random_index]

                    if y_batch[i,0]==1 and id in self.aapm_patient_ids:
                        y_batch[i,1]=1
                        y_batch[i,2]=patch_center[-1]
                    
              
                    x_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                patch_center=patch_center,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_origin=origin,
                                                                world_coord_system=world_coord_system)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break




    


