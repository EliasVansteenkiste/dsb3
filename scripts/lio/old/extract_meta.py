import numpy as np # linear algebra
import dicom
import os
import scipy.ndimage
import dicom
import scipy.stats
import sys
import cPickle
import gzip

import preptools

INPUT_FOLDER_STAGE1 = "/media/lio/Elements/dsb3/stage1/"
OUTPUT_STAGE1 = "/media/lio/Elements/dsb3/meta_stage1.pkl.gz"

INPUT_FOLDER_LUNA = "/media/lio/Elements/dsb3/luna/"
OUTPUT_LUNA = "/media/lio/Elements/dsb3/meta_luna.pkl.gz"


def main():
    extract_stage1()
    # extract_luna()

# def extract_luna():
#

def extract_stage1():
    patients = os.listdir(INPUT_FOLDER_STAGE1)
    patients.sort()
    print len(patients), "patients"

    meta = {}
    for i, patient in enumerate(patients):
        path = INPUT_FOLDER_STAGE1 + patient
        scan = preptools.load_scan(path, True)
        spacing, flipped = preptools.get_spacing(scan[0])
        if flipped: spacing[0] = -spacing[0]
        for s in scan: s.Spacing = spacing
        meta[patient] = scan
        print "%i/%i" % (i + 1, len(patients))
    with gzip.open(OUTPUT_STAGE1, "wb") as f:
        cPickle.dump(meta, f, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()