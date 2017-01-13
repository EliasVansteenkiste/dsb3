import glob
import re
from StringIO import StringIO
import csv
import random

import skimage
import numpy as np

from interfaces.data_loader import StandardDataLoader, TRAINING, VALIDATION, TEST, INPUT, OUTPUT, UNSUPERVISED, compress_data
from utils import paths, varname, put_in_the_middle, put_randomly
import cPickle as pickle
#import h5py
#import bcolz
import time
from scipy.io import loadmat


AUC_TRAINING = "auc training"

class SeizureDataLoader(StandardDataLoader):

    OUTPUT_DATA_SIZE_TYPE = {
        "kaggle-seizure:class":     ((), "uint8"),
        "kaggle-seizure:sample_id": ((), "uint32")
    }

    labels = dict()
    names = dict()

    datasets = [TRAINING, AUC_TRAINING, VALIDATION, TEST]

    def __init__(self, location=paths.DATA_PATH, estimated_labels_path=None, *args, **kwargs):
        super(SeizureDataLoader,self).__init__(location=location, *args, **kwargs)
        self.estimated_labels_path = estimated_labels_path

    def mat_to_data(self, path):
        mat = loadmat(path)
        names = mat['dataStruct'].dtype.names
        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
        return ndata

    def prepare(self):
        #import bcolz
        #bcolz.defaults.out_flavor = "numpy"
        #bcolz.cparams(clevel=9, shuffle=1, cname="blosclz", quantize=1)

        # step 0: load only when not loaded yet

        if TRAINING in self.data \
            and AUC_TRAINING in self.data \
            and VALIDATION in self.data \
            and TEST in self.data:
            return

        print "pre-loading data to RAM"
        try:
            print "loading data as pickle... ",
            self.load_as_pickle()
            print "success!"
            for set in self.datasets:
                print set, len(self.indices[set]), "samples"
            return
        except:
            print "failed!"
        # step 1: load the data
        data_filenames = sorted(glob.glob(self.location+'/train_?/?_*_?.mat'))

        #balance datasets:
        import random
        np.random.seed(317070)
        random.seed(317070)
        random.shuffle(data_filenames)
        # SPEED UP
        data_filenames = data_filenames[:100]
        validation_split = np.random.choice([TRAINING, AUC_TRAINING, VALIDATION], p=[0.8, 0.1, 0.1], replace=True, size=(len(data_filenames),))



        for s in self.datasets:
            self.data[s] = {}
            for k in ["sample_rate", "data", "patient", "sample"]:
                self.data[s][k] = []
            self.labels[s] = []
            self.names[s] = []

        for i, (dataset, data_file_name) in enumerate(zip(validation_split, data_filenames)):
            print "Loading", i+1, "/",len(data_filenames), ":", data_file_name

            I,J,K = data_file_name.split('/')[-1].split('.')[0].split('_')

            data = self.mat_to_data(data_file_name)
            # /2 for downsampling
            self.data[dataset]["sample_rate"].append(data['iEEGsamplingRate'][0,0].astype('float32') / 2)
            #d = bcolz.carray(data['data'][::2,:].T.astype('float16'),
            #                    expectedlen=16*120000,
            #                    #cparams=bcolz.cparams(clevel=9, quantize=0, cname='lz4hc', shuffle=bcolz.NOSHUFFLE))
            #                    cparams=bcolz.cparams(clevel=9, quantize=0, cname='snappy', shuffle=bcolz.NOSHUFFLE))
            d = data['data'].T[:,::2].astype('float16')
            import sys
            self.data[dataset]["data"].append(d)
            self.data[dataset]["patient"].append(int(I))
            self.data[dataset]["sample"].append(int(J))
            self.names[dataset].append(data_file_name)
            self.labels[dataset].append(int(K))

        last_index = -1
        for set in self.datasets:
            self.indices[set] = range(last_index+1,last_index+1+len(self.labels[set]))
            if len(self.indices[set]) > 0:
                last_index = self.indices[set][-1]
            print set, len(self.indices[set]), "samples"
        self.dump_as_pickle()


    #DUMP_FILENAME = 'datadump-micro.pkl'
    DUMP_FILENAME = 'datadump.pkl'
    def dump_as_pickle(self):
        pickle.dump({
            "data": self.data,
            "names": self.names,
            "labels": self.labels,
            "indices": self.indices
        }, open(self.location+self.DUMP_FILENAME, 'wb'),
        protocol=2)

    def load_as_pickle(self):
        d = pickle.load(open(self.location+self.DUMP_FILENAME, 'rb'))
        self.data.update(d["data"])
        self.names.update(d["names"])
        self.labels.update(d["labels"])
        self.indices.update(d["indices"])


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
            if "kaggle-seizure" not in tags:
                continue

            if "default" in tags:
                sample[INPUT][tag] = self.data[set]["data"][sample_index].astype('float32')

            if "sample-rate" in tags:
                sample[INPUT][tag] = np.float32(self.data[set]["sample_rate"][sample_index])

            if "patient" in tags:
                sample[INPUT][tag] = np.float32(self.data[set]["patient"][sample_index])

        for tag in output_keys_to_do:
            tags = tag.split(':')
            if "kaggle-seizure" not in tags:
                continue

            if "class" in tags:
                sample[OUTPUT][tag] = np.int64(self.labels[set][sample_index])

            if "sample_id" in tags:
                sample[OUTPUT][tag] = sample_id

        return sample
