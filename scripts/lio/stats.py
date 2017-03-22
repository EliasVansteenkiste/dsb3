import numpy as np # linear algebra
import dicom
import os
import scipy.ndimage
import preptools
import dicom
import scipy.stats

# Some constants
INPUT_FOLDER = "/media/lio/Elements/dsb3/stage1/"
patients = os.listdir(INPUT_FOLDER)
# patients.sort()
import random
random.shuffle(patients)
print len(patients), "patients"

# spacings = np.empty((len(patients),3))
spacings = []

fixed_spacing = [1, 1, 1]

for i, patient in enumerate(patients):

    path = INPUT_FOLDER + patient
    scan = preptools.load_scan(INPUT_FOLDER+patient, True)
    spacing, flipped = preptools.get_spacing(scan[0])
    # print scan[0]
    # if flipped: image = image[::-1, :, :]

    resize_factor = spacing / fixed_spacing
    new_real_shape = np.array([len(scan),scan[0].Rows, scan[0].Columns]) * resize_factor
    new_shape = np.round(new_real_shape)

    spacings.append(spacing)
    print "%i/%i"%(i+1, len(patients))
    print spacing
    print "Mean spacing:\t", np.mean(spacings, axis=0)
    print "Median spacing:\t", np.median(spacings, axis=0)
    print "5th spacing:\t", np.percentile(spacings, 5, axis=0)
    print "95th spacing:\t", np.percentile(spacings, 95, axis=0)
    print "Std spacing:\t", np.std(spacings, axis=0)
    print scipy.stats.mode(spacings)