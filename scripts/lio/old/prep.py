import os
import preptools
import bcolz
# bcolz.blosc_set_nthreads(1)
import numpy as np
from glob import glob
import SimpleITK as sitk


FIXED_SPACING = [1, 1, 1]
# FIXED_SIZE = [320, 336, 336]
FIXED_SIZE = None # no fixed size


INPUT_FOLDER_STAGE1 = "/media/lio/Elements/dsb3/stage1/"
INPUT_FOLDER_LUNA = "/media/lio/Elements/dsb3/luna/"
if FIXED_SIZE is None:
    OUTPUT_FOLDER = "/media/lio/Elements/dsb3/prep_%i%i%i/" % tuple(FIXED_SPACING)
else:
    OUTPUT_FOLDER = "/media/lio/Elements/dsb3/prep_%i%i%i_%ix%ix%i/" % tuple(FIXED_SPACING + FIXED_SIZE)

if not os.path.exists(OUTPUT_FOLDER): os.mkdir(OUTPUT_FOLDER)


# LUNA

patients = glob(INPUT_FOLDER_LUNA+"*/*.mhd")
patients.sort()
print len(patients), "patients"

for i, patient in enumerate(patients):
    itkimage = sitk.ReadImage(patient)
    image = np.transpose(sitk.GetArrayFromImage(itkimage))
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    pix_resampled, spacing = preptools.resample(image, spacing, FIXED_SPACING, FIXED_SIZE)
    pix_resampled = pix_resampled.transpose((2, 1, 0))[::-1, :, :]
    # if i > -1: preptools.plot_3d(pix_resampled, theshold=-500)
    name = os.path.basename(patient)[:-4]
    data_bcolz = bcolz.carray(array=pix_resampled,
                              chunklen=pix_resampled.shape[0],
                              dtype="int16",
                              cparams=bcolz.cparams(clevel=1, cname="zlib"),  # lz4hc zlib blosc
                              rootdir=OUTPUT_FOLDER + name,
                              mode="w")
    data_bcolz.flush()
    print "%i/%i" % (i + 1, len(patients)), spacing, pix_resampled.shape, pix_resampled.dtype



# STAGE1



patients = os.listdir(INPUT_FOLDER_STAGE1)
patients.sort()
print len(patients), "patients"

for i, patient in enumerate(patients):
    scan = preptools.load_scan(INPUT_FOLDER_STAGE1 + patient)
    pixels = preptools.get_pixels_hu(scan)
    spacing, flipped = preptools.get_spacing(scan[0])
    if flipped: pixels = pixels[::-1, :, :]
    pix_resampled, spacing = preptools.resample(pixels, spacing, FIXED_SPACING, FIXED_SIZE)
    # pix_resampled = pix_resampled.transpose((1,2,0))
    # if i > 0: preptools.plot_3d(pix_resampled, theshold=-500)
    data_bcolz = bcolz.carray(array=pix_resampled,
                              chunklen=pix_resampled.shape[0],
                              dtype="int16",
                              cparams=bcolz.cparams(clevel=1, cname="zlib"),  #lz4hc zlib blosc
                                rootdir=OUTPUT_FOLDER + patient,
                              mode="w")
    data_bcolz.flush()
    print "%i/%i" % (i + 1, len(patients)), patient, pix_resampled.shape





