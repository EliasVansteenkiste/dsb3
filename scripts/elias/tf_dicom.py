import numpy as np
import pathfinder
import csv
import dicom
import os
import re
import numpy as np
import csv
import os
from PIL import Image
from collections import defaultdict
import time

import utils
import pickle



def tf_mhd2pkl(pid):

    mhd_path = pathfinder.LUNA_DATA_PATH + '/' + pid + '.mhd'
    lock_file = '/home/eavsteen/._SimpleITK_lock'
    while os.path.exists(lock_file):
        time.sleep(0.05)
    with open(lock_file, 'w') as outfile:
        outfile.write(str(time.time()))
        import SimpleITK as sitk 
        itk_data = sitk.ReadImage(mhd_path.encode('utf-8'))
        pixel_data = sitk.GetArrayFromImage(itk_data)
        origin = np.array(list(reversed(itk_data.GetOrigin())))
        spacing = np.array(list(reversed(itk_data.GetSpacing())))
    try:
        os.remove(lock_file)
    except OSError:
        pass

    d = {'pixel_data':pixel_data, 'origin':origin, 'spacing':spacing}
    pkl_path = pathfinder.LUNA_DATA_PATH + '/' + pid + '.pkl'
    pickle.dump( d, open( pkl_path, "wb" ) )




train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']
patient_ids = train_pids + valid_pids

patient_paths = [pathfinder.LUNA_DATA_PATH + '/' + p + '.mhd' for p in patient_ids]


for idx, p in enumerate(patient_ids):
    print 'patient', idx, '/', len(patient_paths), p
    tf_mhd2pkl(p)
print pathfinder.LUNA_DATA_PATH





