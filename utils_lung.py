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
    return np.array(d.pixel_array), metadata


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


def get_patient_age(s):
    age = float(s[:-1])
    units = s[-1]
    if units == 'M':
        age /= 12.
    elif units == 'W':
        age /= 52.1429
    return age


def get_patient_data(patient_data_path):
    patient_data = []
    slice_paths = os.listdir(patient_data_path)
    for s in slice_paths:
        slice_id = s.split('.')[0]
        data, metadata = read_dicom(patient_data_path + '/' + s)
        patient_data.append({'data': data, 'metadata': metadata,
                             'slice_id': slice_id})
    return patient_data


def sort_slices(patient_data):
    return sorted(patient_data, key=lambda x: x['metadata']['InstanceNumber'])


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
