import dicom
import SimpleITK as sitk
import numpy as np
import csv
import os
from collections import defaultdict
import cPickle as pickle
import glob
import utils




def histogram_equalization(x, hist=None, bins=None):
    # hist is a normalized histogram, which means that the sum of the counts has to be one
    if hist is None and bins is None:
        bins = np.arange(-950,500,50)
        n_bins = bins.shape[0] -1
        hist = 1. * np.ones(n_bins) / n_bins
    elif hist is None or bins is None:
        raise
    print bins
    assert(len(bins) == (len(hist)+1))

    # init our target array 
    z = np.empty(x.shape)

    # copy the values outside of the bins from the original
    z[x<=bins[0]] = x[x<=bins[0]] 
    z[x>=bins[-1]] = x[x>=bins[-1]] 

    inside_bins = np.logical_and(x>bins[0], x<bins[-1])

    n_bins = bins.shape[0] -1
    prev_percentile = 0
    for i in range(n_bins):
        target_count = hist[i]
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        new_percentile = prev_percentile + target_count*100
        low_orig = np.percentile(x[inside_bins], prev_percentile)
        print new_percentile-100.0
        if i == n_bins-1:
            high_orig = bins[-1]
        else:
            high_orig = np.percentile(x[inside_bins], new_percentile)

        prev_percentile = new_percentile
        elements_to_rescale = np.logical_and(x>=low_orig, x<high_orig)
        y = x[elements_to_rescale]
        y_r = (y - low_orig)/(high_orig-low_orig)*(upper_bound-lower_bound) + lower_bound
        z[elements_to_rescale] = y_r


    return z


def read_pkl(path):
    d = pickle.load(open(path, "rb"))
    return d['pixel_data'], d['origin'], d['spacing']

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


def extract_pid_dir(patient_data_path):
    return patient_data_path.split('/')[-1]


def extract_pid_filename(file_path, replace_str='.mhd'):
    return os.path.basename(file_path).replace(replace_str, '').replace('.pkl', '')


def get_candidates_paths(path):
    id2candidates_path = {}
    file_paths = sorted(glob.glob(path + '/*.pkl'))
    for p in file_paths:
        pid = extract_pid_filename(p, '.pkl')
        id2candidates_path[pid] = p
    return id2candidates_path


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


def ct2HU(x, metadata):
    x = metadata['RescaleSlope'] * x + metadata['RescaleIntercept']
    x[x < -1000] = -1000
    return x


def read_dicom_scan(patient_data_path):
    sid2data, sid2metadata = get_patient_data(patient_data_path)
    sid2position = {}
    for sid in sid2data.keys():
        sid2position[sid] = get_slice_position(sid2metadata[sid])
    sids_sorted = sorted(sid2position.items(), key=lambda x: x[1])
    sids_sorted = [s[0] for s in sids_sorted]
    z_pixel_spacing = []
    for s1, s2 in zip(sids_sorted[1:], sids_sorted[:-1]):
        z_pixel_spacing.append(sid2position[s1] - sid2position[s2])
    z_pixel_spacing = np.array(z_pixel_spacing)
    try:
        assert np.all((z_pixel_spacing - z_pixel_spacing[0]) < 0.01)
    except:
        print 'This patient has multiple series, we will remove one'
        sids_sorted_2 = []
        for s1, s2 in zip(sids_sorted[::2], sids_sorted[1::2]):
            if sid2metadata[s1]["InstanceNumber"] > sid2metadata[s2]["InstanceNumber"]:
                sids_sorted_2.append(s1)
            else:
                sids_sorted_2.append(s2)
        sids_sorted = sids_sorted_2
        z_pixel_spacing = []
        for s1, s2 in zip(sids_sorted[1:], sids_sorted[:-1]):
            z_pixel_spacing.append(sid2position[s1] - sid2position[s2])
        z_pixel_spacing = np.array(z_pixel_spacing)
        assert np.all((z_pixel_spacing - z_pixel_spacing[0]) < 0.01)

    pixel_spacing = np.array((z_pixel_spacing[0],
                              sid2metadata[sids_sorted[0]]['PixelSpacing'][0],
                              sid2metadata[sids_sorted[0]]['PixelSpacing'][1]))

    img = np.stack([ct2HU(sid2data[sid], sid2metadata[sid]) for sid in sids_sorted])

    return img, pixel_spacing


def sort_slices_position(patient_data):
    return sorted(patient_data, key=lambda x: get_slice_position(x['metadata']))


def sort_sids_by_position(sid2metadata):
    return sorted(sid2metadata.keys(), key=lambda x: get_slice_position(sid2metadata[x]))


def sort_slices_jonas(sid2metadata):
    sid2position = slice_location_finder(sid2metadata)
    return sorted(sid2metadata.keys(), key=lambda x: sid2position[x])


def get_slice_position(slice_metadata):
    """
    https://www.kaggle.com/rmchamberlain/data-science-bowl-2017/dicom-to-3d-numpy-arrays
    """
    orientation = tuple((o for o in slice_metadata['ImageOrientationPatient']))
    position = tuple((p for p in slice_metadata['ImagePositionPatient']))
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
    pids = sorted(os.listdir(data_dir))
    return [data_dir + '/' + p for p in pids]

def read_patient_annotations_luna(pid, directory):
    return pickle.load(open(os.path.join(directory,pid+'.pkl'),"rb"))


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


def read_test_labels(file_path):
    id2labels = {}
    train_csv = open(file_path)
    lines = train_csv.readlines()
    i = 0
    for item in lines:
        if i == 0:
            i = 1
            continue
        id, label = item.replace('\n', '').split(';')
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


def write_submission(pid2prediction, submission_path):
    """
    :param pid2prediction: dict of {patient_id: label}
    :param submission_path:
    """
    f = open(submission_path, 'w+')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(['id', 'cancer'])
    for pid in pid2prediction.keys():
        fo.writerow([pid, pid2prediction[pid]])
    f.close()


def filter_close_neighbors(candidates, min_dist=16):
    #TODO pixelspacing should be added , it is now hardcoded 
    candidates_wo_dupes = set()
    no_pairs = 0
    for can1 in candidates:
        found_close_candidate = False
        swap_candidate = None
        for can2 in candidates_wo_dupes:
            if (can1 == can2).all():
                raise "Candidate should not be in the target array yet"
            else:
                delta = can1[:3] - can2[:3]
                delta[0] = 2.5*delta[0] #zyx coos
                dist = np.sum(delta**2)**(1./2)
                if dist<min_dist:
                    no_pairs += 1
                    print 'Warning: there is a pair nodules close together',  can1[:3], can2[:3]
                    found_close_candidate = True
                    if can1[4]>can2[4]:
                        swap_candidate = can2
                    break
        if not found_close_candidate:
            candidates_wo_dupes.add(tuple(can1))
        elif swap_candidate:
            candidates_wo_dupes.remove(swap_candidate)
            candidates_wo_dupes.add(tuple(can1))
    print 'n candidates filtered out', no_pairs
    return candidates_wo_dupes

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


def get_generated_pids(predictions_dir):
    pids = []
    if os.path.isdir(predictions_dir):
        pids = os.listdir(predictions_dir)
        pids = [extract_pid_filename(p) for p in pids]
    return pids
