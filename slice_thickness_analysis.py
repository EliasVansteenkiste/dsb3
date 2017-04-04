import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import utils
import utils_lung
import logger
import sys
import collections
import pathfinder
import data_transforms
import cPickle


def make_slice_thickness_info_file():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_1/'
    utils.auto_make_dir(image_dir)

    sys.stdout = logger.Logger(image_dir + '/%s.log' % 'test1_log')
    sys.stderr = sys.stdout

    patient_data_paths = utils_lung.get_patient_data_paths(pathfinder.DATA_PATH)
    print len(patient_data_paths)

    train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
    train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids[
        'test']

    global pid2slice_info
    pid2slice_info = {}

    def extract_slice_thickness(sid2metadata, sids_sorted):
        try:
            slice_thickness_pos = np.abs(sid2metadata[sids_sorted[0]]['ImagePositionPatient'][2] -
                                         sid2metadata[sids_sorted[1]]['ImagePositionPatient'][2])
        except:
            print 'This patient has no ImagePosition!'
            slice_thickness_pos = 0.

        return slice_thickness_pos

    def calc_amount_slices(sids_sorted):
        return len(sids_sorted)

    def calc_x_scale(sid2metadata, sids_sorted):
        return sid2metadata[sids_sorted[0]]['PixelSpacing'][0]

    def calc_y_scale(sid2metadata, sids_sorted):
        return sid2metadata[sids_sorted[0]]['PixelSpacing'][1]

    def find_dataset_membership(pid):
        if pid in train_pids:
            return 'training set'
        elif pid in valid_pids:
            return 'validation set'
        elif pid in test_pids:
            return 'test set'
        else:
            raise ValueError('Could not find membership of patient % in dataset' % pid)

    count = 0
    for k, p in enumerate(patient_data_paths):
        pid = utils_lung.extract_pid_dir(p)
        try:
            sid2data, sid2metadata = utils_lung.get_patient_data(p)
            sids_sorted = utils_lung.sort_sids_by_position(sid2metadata)
            sids_sorted_jonas = utils_lung.sort_slices_jonas(sid2metadata)
            sid2position = utils_lung.slice_location_finder(sid2metadata)

            slice_thickness = extract_slice_thickness(sid2metadata, sids_sorted)
            amount_slices = calc_amount_slices(sids_sorted)
            x_scale = calc_x_scale(sid2metadata, sids_sorted)
            y_scale = calc_y_scale(sid2metadata, sids_sorted)
            z_scale = slice_thickness
            label = find_dataset_membership(pid)

            slice_info = {
                'slice_thickness': slice_thickness,
                'amount_slices': amount_slices,
                'x_scale': x_scale,
                'y_scale': y_scale,
                'z_scale': z_scale,
                'label': label
            }
            print 'Patient ID "{}" with slice information: {}'.format(pid, slice_info)
            pid2slice_info[pid] = slice_info
        except:
            print 'exception!!!', pid
    cPickle.dump(pid2slice_info, open('/home/adverley/Code/Projects/Kaggle/dsb3/analysis/' + "pid2slice_info.p", "wb"))


pid2slice_info = cPickle.load(open('/home/adverley/Code/Projects/Kaggle/dsb3/analysis/' + "pid2slice_info.p", "r"))

def get_slice_info_of_patient(pid):
    return pid2slice_info[pid]


if __name__ == 'runme':
    # uncomment if file is deleted or want to renew
    # make_slice_thickness_info_file()
    pass
