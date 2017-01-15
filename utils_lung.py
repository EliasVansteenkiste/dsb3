import numpy as np
import pathfinder
import csv
import dicom
import os


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
    metadata['SliceLocation'] = np.float32(metadata['SliceLocation'])
    metadata['ImagePositionPatient'] = np.float32(metadata['ImagePositionPatient'])
    metadata['Rows'] = int(metadata['Rows'])
    metadata['Columns'] = int(metadata['Columns'])
    metadata['RescaleSlope'] = float(metadata['RescaleSlope'])
    metadata['RescaleIntercept'] = float(metadata['RescaleIntercept'])
    return np.array(d.pixel_array), metadata


def get_patient_data(patient_data_path):
    patient_data = []
    pid = patient_data_path.split('/')[-1]
    slice_paths = os.listdir(patient_data_path)
    for s in slice_paths:
        slice_id = s.split('.')[0]
        data, metadata = read_dicom(patient_data_path + '/' + s)
        patient_data.append({'data': data, 'metadata': metadata,
                             'slice_id': slice_id, 'patient_id': pid})
    return patient_data


def sort_slices(patient_data):
    # TODO: maybe make inplace?
    return sorted(patient_data, key=lambda x: x['metadata']['InstanceNumber'])


def sort_slices_slicelocation(patient_data):
    # TODO: maybe make inplace?
    return sorted(patient_data, key=lambda x: x['metadata']['SliceLocation'])


def sort_slices_position(patient_data):
    # TODO: maybe make inplace?
    return sorted(patient_data, key=lambda x: thru_plane_position(x['metadata']))


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
        id2labels[id] = int(float(label))
    return id2labels


def write_submission(patient_predictions, submission_path):
    """
    :param patient_predictions: dict of {patient_id: label}
    :param submission_path:
    """
    fi = csv.reader(open(pathfinder.SAMPLE_SUBMISSION_PATH))
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


if __name__ == "__main__":
    pid2label = read_labels(pathfinder.SAMPLE_SUBMISSION_PATH)
    for k, v in pid2label.iteritems():
        pid2label[k] += 1
    write_submission(pid2label, 'aaa.csv')
