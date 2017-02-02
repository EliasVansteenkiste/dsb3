import os
import bcolz
import numpy as np
from glob import glob
import cPickle
import sys
import gzip
import dicom
import SimpleITK as sitk

SAVE_STAGE1 = False
SAVE_LUNA = False

INPUT_FOLDER_STAGE1 = "/media/lio/Elements/dsb3/stage1/"
INPUT_FOLDER_LUNA = "/media/lio/Elements/dsb3/luna/"
OUTPUT_FOLDER = "/media/lio/Elements/dsb3/_bcolz/"

if not os.path.exists(OUTPUT_FOLDER): os.mkdir(OUTPUT_FOLDER)
if not os.path.exists(OUTPUT_FOLDER+"luna"): os.mkdir(OUTPUT_FOLDER+"luna")
if not os.path.exists(OUTPUT_FOLDER+"stage1"): os.mkdir(OUTPUT_FOLDER+"stage1")

meta = {}

# LUNA
def compress_luna():
    global metadata

    patients = glob(INPUT_FOLDER_LUNA+"*/*.mhd")
    patients.sort()
    assert len(patients) == 888

    for i, patient in enumerate(patients):
        name = os.path.basename(patient)[:-4]
        print "%i/%i" % (i + 1, len(patients)), name
        itkimage = sitk.ReadImage(patient)
        meta[name] = {"origin": np.array(itkimage.GetOrigin()),
                      "pixelspacing": np.array(itkimage.GetSpacing())}
        if SAVE_LUNA:
            save_bcolz(sitk.GetArrayFromImage(itkimage),
                   OUTPUT_FOLDER + "luna/" + name)

# STAGE1
def compress_stage1():
    global metadata

    patients = os.listdir(INPUT_FOLDER_STAGE1)
    patients.sort()
    assert len(patients) == 1595

    for i, patient in enumerate(patients):
        print "%i/%i" % (i + 1, len(patients)), patient
        path = INPUT_FOLDER_STAGE1+patient
        slices = [dicom.read_file(path + '/' + s, stop_before_pixels=SAVE_STAGE1) for s in os.listdir(path)]
        meta[patient] = {
            "imagepositions": [s.ImagePositionPatient for s in slices],
            "rescaleintercept": [s.RescaleIntercept for s in slices],
            "rescaleslope": [s.RescaleSlope for s in slices],
            "pixelspacing": slices[0].PixelSpacing
        }
        if SAVE_STAGE1:
            save_bcolz(np.stack([s.pixel_array for s in slices]).astype(np.int16),
                       OUTPUT_FOLDER + "stage1/" + patient)


def save_meta():
    global metadata

    with gzip.open(OUTPUT_FOLDER + "metadata.pkl.gz", "wb") as f:
        cPickle.dump(meta, f, protocol=cPickle.HIGHEST_PROTOCOL)


def save_bcolz(data, rootdir):
    data_bcolz = bcolz.carray(array=data,
                              chunklen=data.shape[0],
                              dtype="int16",
                              cparams=bcolz.cparams(clevel=1, cname="zlib"),  # lz4hc zlib blosc
                              rootdir=rootdir,
                              mode="w")
    data_bcolz.flush()


if __name__ == '__main__':
    compress_stage1()
    compress_luna()
    save_meta()