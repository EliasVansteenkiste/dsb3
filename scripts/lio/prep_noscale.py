import os
import preptools
import bcolz
# bcolz.blosc_set_nthreads(1)
import numpy as np
from glob import glob
import cPickle
import sys
import gzip
import SimpleITK as sitk
from utils import paths


INPUT_FOLDER_STAGE1 = paths.DATA_PATH
INPUT_FOLDER_LUNA = paths.LUNA_DATA_PATH
OUTPUT_FOLDER = paths.INTERMEDIATE_DATA_PATH
DATA_FOLDER = paths.ALL_DATA_PATH

if not os.path.exists(OUTPUT_FOLDER): os.mkdir(OUTPUT_FOLDER)

spacings = {}
origins = {}

# LUNA

patients = glob(INPUT_FOLDER_LUNA+"*.mhd")
patients.sort()
print len(patients), "patients"
for i, patient in enumerate(patients):
    itkimage = sitk.ReadImage(patient.encode('utf8'))
    image = sitk.GetArrayFromImage(itkimage)
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    image = image[::-1,:,:]  # to match the kaggle data order
    spacing = spacing  # go from XYZ to XYZ
    origin = origin  # go from XYZ to XYZ
    # if i > -1: preptools.plot_3d(image, theshold=-500, spacing=spacing)
    name = os.path.basename(patient)[:-4]
    spacings[name] = tuple(spacing)
    origins[name] = tuple(origin)
    # data_bcolz = bcolz.carray(array=image,
    #                           chunklen=image.shape[0],
    #                           dtype="int16",
    #                           cparams=bcolz.cparams(clevel=1, cname="zlib"),  # lz4hc zlib blosc
    #                           rootdir=DATA_FOLDER + name,
    #                           mode="w")
    # data_bcolz.flush()
    print "%i/%i" % (i + 1, len(patients)), spacing
    # sys.exit()

with gzip.open(OUTPUT_FOLDER+"spacings.pkl.gz", "wb") as f:
    cPickle.dump(spacings, f, protocol=cPickle.HIGHEST_PROTOCOL)

with gzip.open(OUTPUT_FOLDER+"origins.pkl.gz", "wb") as f:
    cPickle.dump(origins, f, protocol=cPickle.HIGHEST_PROTOCOL)

# STAGE1


patients = os.listdir(INPUT_FOLDER_STAGE1)
patients.sort()
print len(patients), "patients"

for i, patient in enumerate(patients):
    scan = preptools.load_scan(INPUT_FOLDER_STAGE1 + patient)
    # scan = preptools.load_scan(INPUT_FOLDER_STAGE1 + patient, stop_before_pixels=True)
    pixels = preptools.get_pixels_hu(scan)
    spacing, flipped = preptools.get_spacing(scan[0])
    spacings[patient] = tuple(spacing)
    if flipped: pixels = pixels[::-1, :, :]
    # if i > -1: preptools.plot_3d(pixels, theshold=-500, spacing=spacing)
    data_bcolz = bcolz.carray(array=pixels,
                              chunklen=pixels.shape[0],
                              dtype="int16",
                              cparams=bcolz.cparams(clevel=1, cname="zlib"),  #lz4hc zlib blosc
                                rootdir=DATA_FOLDER + patient,
                              mode="w")
    data_bcolz.flush()
    print "%i/%i" % (i + 1, len(patients)), patient, spacing#, pixels.shape


with gzip.open(OUTPUT_FOLDER+"spacings.pkl.gz", "wb") as f:
    cPickle.dump(spacings, f, protocol=cPickle.HIGHEST_PROTOCOL)