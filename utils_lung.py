import numpy as np
import csv
import dicom
import os
import re
import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
from collections import defaultdict


def read_mhd(path):
    itk_data = sitk.ReadImage(path.encode('utf-8'))
    pixel_data = sitk.GetArrayFromImage(itk_data)
    origin = np.array(list(reversed(itk_data.GetOrigin())))
    spacing = np.array(list(reversed(itk_data.GetSpacing())))
    return pixel_data, origin, spacing


def world2voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord


def read_dicom(path):
    d = dicom.read_file(path)
    metadata = {}
    for attr in dir(d):
        if attr[0].isupper() and attr != 'PixelData':
            try:
                metadata[attr] = getattr(d, attr)
            except AttributeError:
                pass

    metadata['InstanceNumber'] = int(metadata['InstanceNumber'])
    metadata['PixelSpacing'] = np.float32(metadata['PixelSpacing'])
    metadata['ImageOrientationPatient'] = np.float32(metadata['ImageOrientationPatient'])
    try:
        metadata['SliceLocation'] = np.float32(metadata['SliceLocation'])
    except:
        metadata['SliceLocation'] = None
    metadata['ImagePositionPatient'] = np.float32(metadata['ImagePositionPatient'])
    metadata['Rows'] = int(metadata['Rows'])
    metadata['Columns'] = int(metadata['Columns'])
    metadata['RescaleSlope'] = float(metadata['RescaleSlope'])
    metadata['RescaleIntercept'] = float(metadata['RescaleIntercept'])
    return np.array(d.pixel_array), metadata


def extract_pid(patient_data_path):
    return patient_data_path.split('/')[-1]


def luna_extract_pid(patient_data_path, replace_str='.mhd'):
    return os.path.basename(patient_data_path).replace(replace_str, '')


def get_patient_data(patient_data_path):
    slice_paths = os.listdir(patient_data_path)
    sid2data = {}
    sid2metadata = {}
    for s in slice_paths:
        slice_id = s.split('.')[0]
        data, metadata = read_dicom(patient_data_path + '/' + s)
        sid2data[slice_id] = data
        sid2metadata[slice_id] = metadata
    return sid2data, sid2metadata


def sort_slices_intance_number(sid2metadata):
    return sorted(sid2metadata.keys(), key=lambda x: sid2metadata[x]['InstanceNumber'])


def sort_slices_position(patient_data):
    return sorted(patient_data, key=lambda x: thru_plane_position(x['metadata']))


def sort_slices_plane(sid2metadata):
    return sorted(sid2metadata.keys(), key=lambda x: thru_plane_position(sid2metadata[x]))


def sort_slices_jonas(sid2metadata):
    sid2position = slice_location_finder(sid2metadata)
    return sorted(sid2metadata.keys(), key=lambda x: sid2position[x])


def thru_plane_position(slice_metadata):
    """
    https://www.kaggle.com/rmchamberlain/data-science-bowl-2017/dicom-to-3d-numpy-arrays
    """
    orientation = tuple((float(o) for o in slice_metadata['ImageOrientationPatient']))
    position = tuple((float(p) for p in slice_metadata['ImagePositionPatient']))
    rowvec, colvec = orientation[:3], orientation[3:]
    normal_vector = np.cross(rowvec, colvec)
    slice_pos = np.dot(position, normal_vector)
    return slice_pos


def slice_location_finder(sid2metadata):
    """
    :param slicepath2metadata: dict with arbitrary keys, and metadata values
    :return:
    """

    sid2midpix = {}
    sid2position = {}

    for sid in sid2metadata:
        metadata = sid2metadata[sid]
        image_orientation = metadata["ImageOrientationPatient"]
        image_position = metadata["ImagePositionPatient"]
        pixel_spacing = metadata["PixelSpacing"]
        rows = metadata['Rows']
        columns = metadata['Columns']

        # calculate value of middle pixel
        F = np.array(image_orientation).reshape((2, 3))
        # reversed order, as per http://nipy.org/nibabel/dicom/dicom_orientation.html
        i, j = columns / 2.0, rows / 2.0
        im_pos = np.array([[i * pixel_spacing[0], j * pixel_spacing[1]]], dtype='float32')
        pos = np.array(image_position).reshape((1, 3))
        position = np.dot(im_pos, F) + pos
        sid2midpix[sid] = position[0, :]

    if len(sid2midpix) <= 1:
        for sp, midpix in sid2midpix.iteritems():
            sid2position[sp] = 0.
    else:
        # find the keys of the 2 points furthest away from each other
        max_dist = -1.0
        max_dist_keys = []
        for sp1, midpix1 in sid2midpix.iteritems():
            for sp2, midpix2 in sid2midpix.iteritems():
                if sp1 == sp2:
                    continue
                distance = np.sqrt(np.sum((midpix1 - midpix2) ** 2))
                if distance > max_dist:
                    max_dist_keys = [sp1, sp2]
                    max_dist = distance
        # project the others on the line between these 2 points
        # sort the keys, so the order is more or less the same as they were
        # max_dist_keys.sort(key=lambda x: int(re.search(r'/sax_(\d+)\.pkl$', x).group(1)))
        p_ref1 = sid2midpix[max_dist_keys[0]]
        p_ref2 = sid2midpix[max_dist_keys[1]]
        v1 = p_ref2 - p_ref1
        v1 /= np.linalg.norm(v1)

        for sp, midpix in sid2midpix.iteritems():
            v2 = midpix - p_ref1
            sid2position[sp] = np.inner(v1, v2)

    return sid2position


def get_patient_data_paths(data_dir):
    pids = os.listdir(data_dir)
    return [data_dir + '/' + p for p in pids]


def read_labels(file_path):
    id2labels = {}
    train_csv = open(file_path)
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, label = item.replace('\n', '').split(',')
        id2labels[id] = int(label)
    return id2labels


def read_luna_annotations(file_path):
    id2xyzd = defaultdict(list)
    train_csv = open(file_path)
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, x, y, z, d = item.replace('\n', '').split(',')
        id2xyzd[id].append([float(z), float(y), float(x), float(d)])
    return id2xyzd


def read_luna_negative_candidates(file_path):
    id2xyzd = defaultdict(list)
    train_csv = open(file_path)
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, x, y, z, d = item.replace('\n', '').split(',')
        if float(d) == 0:
            id2xyzd[id].append([float(z), float(y), float(x), float(d)])
    return id2xyzd


def write_submission(patient_predictions, submission_path):
    """
    :param patient_predictions: dict of {patient_id: label}
    :param submission_path:
    """
    fi = csv.reader(open(submission_path))
    f = open(submission_path, 'w+')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        pid = line[0]
        if pid in patient_predictions.keys():
            fo.writerow([pid, patient_predictions[pid]])
        else:
            print 'missed patient:', pid
    f.close()


def dice_index(predictions, targets, epsilon=1e-12):
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    dice = (2. * np.sum(targets * predictions) + epsilon) / (np.sum(predictions) + np.sum(targets) + epsilon)
    return dice


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.asarray(predictions).flatten()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    targets = np.asarray(targets).flatten()
    ce = np.mean(np.log(predictions) * targets + np.log(1 - predictions) * (1. - targets))
    return ce

# def luna_merge_negative_candidates(file_path1, file_path2, output_file_path):
#     candidates_csv1 = csv.reader(open(file_path1))
#     candidates_csv2 = csv.reader(open(file_path2))
#
#     f = open(output_file_path, 'w+')
#     fo = csv.writer(f, lineterminator='\n')
#
#     fo.writerow(candidates_csv1.next())
#     for line in candidates_csv1:
#         label = int(line[-1])
#         if label == 0:
#             fo.writerow(line)
#
#     candidates_csv2.next()
#     for line in candidates_csv2:
#         label = int(line[-1])
#         if label == 0:
#             fo.writerow(line)
#     f.close()
