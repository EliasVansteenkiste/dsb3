from collections import defaultdict
import glob
import csv
import gzip
import os
import random
from os import path
import bcolz
import cPickle
from multiprocessing import Lock
import numpy as np
import sys
from application.bcolz_all_data import load_luna_labels

from interfaces.data_loader import StandardDataLoader, TRAINING, VALIDATION, TEST, INPUT, OUTPUT, TRAIN
from utils import paths
from utils.custom_warnings import deprecated


VALIDATION_SET_SIZE = 0.2

"""
This class is responsible for loading the data in your config

It extends from StandardDataLoader, which does the fancy stuff, like being deterministic, loading and preprocessing multithreaded,
The data loader is first prepared, and load_sample then returns data for each requested tag for a specific patient

Note that every patient, independent on its set, gets a unique number.
Every time that number is requested, exactly the same data needs to be returned
"""
class LunaDataLoader(StandardDataLoader):

    # These are shared between all objects of this type
    labels = dict()
    names = dict()

    datasets = [TRAIN, VALIDATION]

    def __init__(self, location=paths.LUNA_DATA_PATH, only_positive=False,pick_nodule=False, *args, **kwargs):
        super(LunaDataLoader,self).__init__(location=location, *args, **kwargs)
        self.only_positive = only_positive
        self.pick_nodule=pick_nodule

    def prepare(self):
        """
        Prepare the dataloader, by storing values to static fields of this class
        In this case, only filenames are loaded prematurely
        :return:
        """

        # step 0: load only when not loaded yet
        if TRAINING in self.data \
            and VALIDATION in self.data:
            return

        # step 1: load the file names
        file_list = sorted(glob.glob(self.location+"*.mhd"))
        # count the number of data points

        # make a stratified validation set
        # note, the seed decides the validation set, but it is deterministic in the names
        random.seed(317070)
        patient_names = [self.patient_name_from_file_name(f) for f in file_list]
        validation_patients = random.sample(patient_names, int(VALIDATION_SET_SIZE*len(patient_names)))

        # make the static data empty
        for s in self.datasets:
            self.data[s] = []
            self.labels[s] = []
            self.names[s] = []

        # load the filenames and put into the right dataset
        labels_as_dict = defaultdict(list)

        with open(paths.LUNA_LABELS_PATH, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip the header
            for row in reader:
                label = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
                labels_as_dict[str(row[0])].append(label)

        for patient_file in file_list:
            patient_name = self.patient_name_from_file_name(patient_file)

            if patient_name in validation_patients:
                s = VALIDATION
            else:
                s = TRAINING
            label = labels_as_dict[str(patient_name)]
            if self.only_positive and not label:
                continue
            self.data[s].append(patient_file)
            
            if self.pick_nodule:
                self.labels[s].append([random.choice(label)])                
            else:
                self.labels[s].append(label)
            
                
            self.names[s].append(patient_name)

        # give every patient a unique number
        last_index = -1
        for s in self.datasets:
            self.indices[s] = range(last_index+1,last_index+1+len(self.data[s]))
            if len(self.indices[s]) > 0:
                last_index = self.indices[s][-1]
            print s, len(self.indices[s]), "samples"

        

    @staticmethod
    def patient_name_from_file_name(patient_file):
        return os.path.splitext(os.path.basename(patient_file))[0]


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


        patientdata = self.load_patient_data(self.data[set][sample_index])

        # Iterate over input tags and return a dict with the requested tags filled
        for tag in input_keys_to_do:
            tags = tag.split(':')
            if "luna" not in tags:
                continue
            if "filename" in tags:
                sample[INPUT][tag] = self.names[set][sample_index]

            if "patient_id" in tags:
                sample[INPUT][tag] = self.names[set][sample_index]

            if "3d" in tags or "default" in tags:
                sample[INPUT][tag] = patientdata["pixeldata"].astype('float32')

            if "pixelspacing" in tags:
                sample[INPUT][tag] = patientdata["spacing"].astype('float32')  # in mm per pixel

            if "shape" in tags:
                sample[INPUT][tag] = patientdata["pixeldata"].shape

            if "origin" in tags:
                sample[INPUT][tag] = patientdata["origin"]

            if "labels" in tags:
                sample[INPUT][tag] = self.labels[set][sample_index]

        for tag in output_keys_to_do:
            tags = tag.split(':')
            if "luna" not in tags:
                continue
            if "sample_id" in tags:
                sample[OUTPUT][tag] = sample_id

            if "segmentation" in tags:
                sample[OUTPUT][tag] =  self.generate_mask(
                    self.labels[set][sample_index],
                    origin = patientdata["origin"],
                    spacing = patientdata["spacing"],
                    shape = patientdata["pixeldata"].shape
                )

            if "gaussian" in tags:
                sample[OUTPUT][tag] =  self.generate_gaussian_mask(
                    self.labels[set][sample_index],
                    origin = patientdata["origin"],
                    spacing = patientdata["spacing"],
                    shape = patientdata["pixeldata"].shape
                )

        return sample


    def load_patient_data(self, path):
        result = dict()
        pixel_data, origin, spacing = self.read_mhd_file(path)
        result["pixeldata"] = pixel_data.T  # move from zyx to xyz
        result["origin"] = origin  # move from zyx to xyz
        result["spacing"] = spacing  # move from zyx to xyz
        return result


    _SimpleITKLock = Lock()
    @staticmethod
    def read_mhd_file(path):
        # SimpleITK has trouble with multiprocessing :-/
        with LunaDataLoader._SimpleITKLock:

            import SimpleITK as sitk    # sudo pip install --upgrade pip; sudo pip install SimpleITK
            itk_data = sitk.ReadImage(path.encode('utf-8'))
            pixel_data = sitk.GetArrayFromImage(itk_data)
            origin = np.array(list(itk_data.GetOrigin()))
            spacing = np.array(list(itk_data.GetSpacing()))
        return pixel_data, origin, spacing


    @staticmethod
    def world_to_voxel_coordinates(world_coord, origin, spacing):
        # TODO: this np.absolute is so weird....
        stretched_voxel_coord = np.absolute(world_coord - origin)
        voxel_coord = stretched_voxel_coord / spacing
        return voxel_coord


    def generate_mask(self, labels, origin, spacing, shape):
        mask = np.zeros(shape=shape, dtype='float32')
        x,y,z = np.ogrid[:mask.shape[0],:mask.shape[1],:mask.shape[2]]

        for label in labels:
            position = np.array(label[:3])
            diameter_in_mm = label[3]
            xt, yt, zt = self.world_to_voxel_coordinates(position, origin, spacing)
            # some basic checks! Very useful in finding errors in the labels
            assert (0<=xt<=shape[0]),xt
            assert (0<=yt<=shape[1]),yt
            assert (0<=zt<=shape[2]),zt
            distance2 = ((spacing[0]*(x-xt))**2 + (spacing[1]*(y-yt))**2 + (spacing[2]*(z-zt))**2)
            mask[(distance2 <= (diameter_in_mm/2.0)**2)] = 1
        return mask


    def generate_gaussian_mask(self, labels, origin, spacing, shape):
        mask = np.zeros(shape=shape, dtype='float32')
        x,y,z = np.ogrid[:mask.shape[0],:mask.shape[1],:mask.shape[2]]
        for label in labels:
            position = np.array(label[:3])
            diameter_in_mm = label[3]
            xt, yt, zt = self.world_to_voxel_coordinates(position, origin, spacing)
            # some basic checks! Very useful in finding errors in the labels
            assert (0<=xt<=shape[0]),xt
            assert (0<=yt<=shape[1]),yt
            assert (0<=zt<=shape[2]),zt
            distance_in_mm2 = ((spacing[0]*(x-xt))**2 + (spacing[1]*(y-yt))**2 + (spacing[2]*(z-zt))**2)
            gaussian = np.exp(- 1.*distance_in_mm2 / (2*diameter_in_mm**2))
            mask += gaussian
        mask = mask/np.max(mask)
        return mask



class BcolzLunaDataLoader(LunaDataLoader):

    # These are shared between all objects of this type
    spacings = dict()
    origins = dict()

    def __init__(self, location=paths.ALL_DATA_PATH, *args, **kwargs):
        super(BcolzLunaDataLoader,self).__init__(location=location, *args, **kwargs)

    def prepare(self):
        """
        Prepare the dataloader, by storing values to static fields of this class
        In this case, only filenames are loaded prematurely
        :return:
        """
        bcolz.set_nthreads(2)

        # step 0: load only when not loaded yet
        if TRAINING in self.data and VALIDATION in self.data: return

        # step 1: load the file names
        patients = sorted(glob.glob(self.location+'/*.*/'))
        print len(patients), "patients"

        # step 1: load the file names
        # make a stratified validation set
        # note, the seed decides the validation set, but it is deterministic in the names
        random.seed(317070)
        patient_names = [self.patient_name_from_file_name(f) for f in patients]
        validation_patients = random.sample(patient_names, int(VALIDATION_SET_SIZE*len(patient_names)))

        labels_as_dict = defaultdict(list)

        with open(paths.LUNA_LABELS_PATH, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip the header
            for row in reader:
                label = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
                labels_as_dict[str(row[0])].append(label)

        # make the static data empty
        for s in self.datasets:
            self.data[s] = []
            self.labels[s] = []
            self.names[s] = []
            self.spacings[s] = []
            self.origins[s] = []

        with gzip.open(paths.INTERMEDIATE_DATA_PATH + 'spacings.pkl.gz') as f:
            spacings = cPickle.load(f)

        with gzip.open(paths.INTERMEDIATE_DATA_PATH + 'origins.pkl.gz') as f:
            origins = cPickle.load(f)

        # load the filenames and put into the right dataset
        for i, patient_folder in enumerate(patients):
            patient_id = str(patient_folder.split(path.sep)[-2])
            if patient_id in validation_patients:
                dataset = VALIDATION
            else:
                dataset = TRAIN


            label = labels_as_dict[patient_id]
            if self.only_positive and not label:
                continue

            self.data[dataset].append(patient_folder)
            self.labels[dataset].append(label)
            self.names[dataset].append(patient_id)
            self.spacings[dataset].append(spacings[patient_id])
            self.origins[dataset].append(origins[patient_id])

        # give every patient a unique number
        last_index = -1
        for set in self.datasets:
            self.indices[set] = range(last_index+1,last_index+1+len(self.data[set]))
            if len(self.indices[set]) > 0:
                last_index = self.indices[set][-1]
            print set, len(self.indices[set]), "samples"

    @staticmethod
    def patient_name_from_file_name(patient_file):
        return os.path.split(os.path.dirname(patient_file))[1].encode('utf8')


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
            if "luna" not in tags: continue

            if "filename" in tags:
                sample[INPUT][tag] = patient_name

            if "patient_id" in tags:
                sample[INPUT][tag] = patient_name

            if "3d" in tags or "default" in tags:
                sample[INPUT][tag] = volume[::-1,:,:]  # see prep_noscale

            if "pixelspacing" in tags:
                sample[INPUT][tag] = self.spacings[set][sample_index]  # in mm per pixel

            if "origin" in tags:
                sample[INPUT][tag] = np.array(list(self.origins[set][sample_index]))

            if "shape" in tags:
                sample[INPUT][tag] = volume.shape

        for tag in output_keys_to_do:
            tags = tag.split(':')
            if "luna" not in tags: continue

            if "target" in tags:
                sample[OUTPUT][tag] = np.int64(self.labels[set][sample_index])

            if "sample_id" in tags:
                sample[OUTPUT][tag] = sample_id

            if "segmentation" in tags:
                sample[OUTPUT][tag] = self.generate_mask(
                    labels=self.labels[set][sample_index],
                    origin=self.origins[set][sample_index],
                    spacing=self.spacings[set][sample_index],
                    shape=volume.shape
                )

            if "gaussian" in tags:
                sample[OUTPUT][tag] = self.generate_gaussian_mask(
                    labels=self.labels[set][sample_index],
                    origin=self.origins[set][sample_index],
                    spacing=self.spacings[set][sample_index],
                    shape=volume.shape
                )

        return sample


class OnlyPositiveLunaDataLoader(LunaDataLoader):
    """
    This dataloader will only return samples which do contain a positive segmentation!

    """
    @deprecated
    def __init__(self, *args, **kwargs):
        super(OnlyPositiveLunaDataLoader,self).__init__(only_positive=True, *args, **kwargs)


# class FPRLunaDataLoader(LunaDataLoader):
#     candidates = dict()
#
#     def __init__(self, candidates_path, *args, **kwargs):
#         super(FPRLunaDataLoader,self).__init__(*args, **kwargs)
#         self.candidates_path = candidates_path
#
#     def prepare(self):
#         super(FPRLunaDataLoader, self).prepare()
#
#         # make the static data empty
#         for s in self.datasets:
#             self.candidates[s] = []
#
#         candidates = defaultdict(lambda: defaultdict(list))
#
#         with open(self.candidates_path, 'rb') as csvfile:
#             reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#             next(reader)  # skip the header
#             for row in reader:
#                 c = (float(row[1]), float(row[2]), float(row[3]), int(row[4]))
#                 candidates[str(row[0])][int(row[4])].append(c)
#
#         for set, patient_files in self.data.items():
#             for patient_file in patient_files:
#                 patient_name = self.patient_name_from_file_name(patient_file)
#                 self.candidates[set].append(candidates[str(patient_name)])
#
#     def load_sample(self, sample_id, input_keys_to_do, output_keys_to_do):
#
#         # find which set this sample is in
#         set, set_indices = None, None
#         for set, set_indices in self.indices.iteritems():
#             if sample_id in set_indices:
#                 break
#
#         sample_id =
#
#         super(FPRLunaDataLoader, self).load_sample(sample_id, input_keys_to_do, output_keys_to_do)
