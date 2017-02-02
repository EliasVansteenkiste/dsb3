import os
import bcolz
import numpy as np
from glob import glob
import cPickle
import sys
import gzip
import dicom
import SimpleITK as sitk

from utils import paths


def mkdir_or_exists(path):
    if not os.path.exists(path): os.mkdir(path)

mkdir_or_exists(paths.BCOLZ_DATA_PATH)
mkdir_or_exists(paths.BCOLZ_DATA_PATH+"luna")
mkdir_or_exists(paths.BCOLZ_DATA_PATH+"stage1")

# Luna: "origin", "pixelspacing"
# Stage1: "imagepositions", "pixelspacing", "rescaleintercept", "rescaleslope"
metadata = {}


# LUNA
def compress_luna():
    global metadata

    patients = glob(paths.LUNA_DATA_PATH+"*/*.mhd")
    patients.sort()
    assert len(patients) == 888

    for i, patient in enumerate(patients):
        name = os.path.basename(patient)[:-4]
        print "%i/%i" % (i + 1, len(patients)), name
        itkimage = sitk.ReadImage(patient)
        metadata[name] = {"origin": np.array(itkimage.GetOrigin()),
                      "pixelspacing": np.array(itkimage.GetSpacing())}
        save_bcolz(sitk.GetArrayFromImage(itkimage),
            paths.BCOLZ_DATA_PATH + "luna/" + name)

# STAGE1
def compress_stage1():
    global metadata

    patients = os.listdir(paths.DATA_PATH)
    patients.sort()
    assert len(patients) == 1595

    for i, patient in enumerate(patients):
        print "%i/%i" % (i + 1, len(patients)), patient
        path = paths.DATA_PATH+patient
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        metadata[patient] = {
            "imagepositions": [s.ImagePositionPatient for s in slices],
            "rescaleintercept": [s.RescaleIntercept for s in slices],
            "rescaleslope": [s.RescaleSlope for s in slices],
            "pixelspacing": slices[0].PixelSpacing
        }
        save_bcolz(np.stack([s.pixel_array for s in slices]).astype(np.int16),
                   paths.BCOLZ_DATA_PATH + "stage1/" + patient)


def save_meta():
    global metadata
    with gzip.open(paths.BCOLZ_DATA_PATH + "metadata.pkl.gz", "wb") as f:
        cPickle.dump(metadata, f, protocol=cPickle.HIGHEST_PROTOCOL)


def save_bcolz(data, rootdir):
    data_bcolz = bcolz.carray(array=data,
                              chunklen=data.shape[0],
                              dtype="int16",
                              cparams=bcolz.cparams(clevel=1, cname="zlib"),  # lz4hc zlib blosc
                              rootdir=rootdir,
                              mode="w")
    data_bcolz.flush()


if __name__ == '__main__':
    compress_luna()
    compress_stage1()
    save_meta()