import glob
import csv
import random
from os import path
import dicom

import numpy as np

from interfaces.data_loader import StandardDataLoader, TRAINING, VALIDATION, TEST, INPUT, OUTPUT, TRAIN
from utils import paths

VALIDATION_SET_SIZE = 0.2

class PatientDataLoader(StandardDataLoader):

    def filter_samples(self):
        pass

    OUTPUT_DATA_SIZE_TYPE = {
        "kaggle-dsb3:class":     ((), "uint8"),
        "kaggle-dsb3:sample_id": ((), "uint32")
    }

    # These are shared between all objects of this type
    labels = dict()
    names = dict()

    datasets = [TRAIN, VALIDATION, TEST]

    def __init__(self, location=paths.DATA_PATH, *args, **kwargs):
        super(PatientDataLoader,self).__init__(location=location, *args, **kwargs)

    def prepare(self):
        # step 0: load only when not loaded yet
        if TRAINING in self.data \
            and VALIDATION in self.data \
            and TEST in self.data:
            return

        # step 1: load the file names
        patients = sorted(glob.glob(self.location+'/*/'))

        labels = dict()
        with open(paths.LABELS_PATH, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip the header
            for row in reader:
                labels[row[0]] = int(row[1])

        random.seed(317070)
        ids_per_label = [[patient_id for patient_id,label in labels.iteritems() if label==l] for l in [0,1]]
        validation_patients = sum([random.sample(sorted(ids), int(VALIDATION_SET_SIZE*len(ids))) for ids in ids_per_label],[])

        for s in self.datasets:
            self.data[s] = []
            self.labels[s] = []
            self.names[s] = []

        for i, patient_folder in enumerate(patients):
            patient_id = patient_folder.split(path.sep)[-2]
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

        sample = dict()
        sample[INPUT] = dict()
        sample[OUTPUT] = dict()

        # Iterate over input tags
        for tag in input_keys_to_do:
            tags = tag.split(':')
            if "dsb3" not in tags:
                continue

            if "filename" in tags:
                sample[INPUT][tag] = self.data[set][sample_index]

            if "all" in tags:
                sample[INPUT][tag] = self.load_patient_data(self.data[set][sample_index])

            if "3d" in tags:
                sample[INPUT][tag] = self.get_raw_3d_data(self.data[set][sample_index]).astype('float32')

            if "default" in tags:
                sample[INPUT][tag] = self.data[set]["data"][sample_index].astype('float32')

        for tag in output_keys_to_do:
            tags = tag.split(':')
            if "dsb3" not in tags:
                continue

            if "class" in tags:
                sample[OUTPUT][tag] = np.int64(self.labels[set][sample_index])

            if "sample_id" in tags:
                sample[OUTPUT][tag] = sample_id

        return sample


    def get_raw_3d_data(self, path):
        slices = self.load_patient_data(path)
        d = []
        for sl in slices.itervalues():
            d.append( (sl["InstanceNumber"], sl["PixelData"]) )
        d.sort()
        result = np.stack([item[1] for item in d])
        return result


    def load_patient_data(self, path):
        images = sorted(glob.glob(path+'*.dcm'))
        result = dict()
        for image in images:
            result[image] = self.read_dicom(image)

        # remove the ones which errored
        result = dict((k, v) for k, v in result.iteritems() if v)
        return result


    def read_dicom(self,filename):
        d = dicom.read_file(filename, force=True)
        data = {}
        try:
            for attr in dir(d):
                if attr[0].isupper() and attr != 'PixelData':
                    try:
                        data[attr] = getattr(d, attr)
                    except AttributeError:
                        pass
                data["PixelData"] = np.array(d.pixel_array)
            data = self.clean_dicom_data(data)
        except:
            print "Failed to load the data in %s" % filename
            return None
        return data


    def clean_dicom_data(self, data):
        for key,value in data.iteritems():
            try:
                if key == 'AcquisitionNumber':
                    data[key] = int(value) if value != '' else value
                elif key == 'BitsAllocated' or key=="BitsStored":
                    data[key] = int(value)
                elif key == 'BurnedInAnnotation':
                    data[key] = False if value=="NO" else True
                elif key == "Columns":
                    data[key] = int(value)
                elif key == "HighBit":
                    data[key] = int(value)
                elif key == "ImageOrientationPatient" or key=="ImagePositionPatient":
                    data[key] = [float(v) for v in value]
                elif key == "InstanceNumber":
                    data[key] = int(value)
                elif key == "PixelData":
                    pass
                elif key == 'PixelRepresentation':
                    data[key] = int(value)
                elif key == "PixelSpacing":
                    data[key] = [float(v) for v in value]
                elif key == 'RescaleIntercept' or key == "RescaleSlope" or key == "Rows":
                    data[key] = int(value)
                elif key == 'SamplesPerPixel':
                    data[key] = int(value)
                elif key == 'SeriesNumber':
                    data[key] = int(value)
                elif key == 'SliceLocation':
                    data[key] = float(value)
                elif key == 'WindowCenter' or key == 'WindowWidth':
                    try:
                        data[key] = [int(v) for v in value]
                    except:
                        data[key] = int(value)
                else:
                    data[key] = str(value)
            except:
                print "Could not clean key %s with value %s"%(key,value)
        return data


