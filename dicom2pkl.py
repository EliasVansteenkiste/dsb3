import os
import cPickle as pickle
import utils
import dicom
import numpy as np


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


def save_data(data, metadata, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump({'data': data,
                     'metadata': metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print 'saved to %s' % out_path


def convert_patient_dicom2pkl(in_path, out_path):
    print in_path
    in_img_paths = os.listdir(in_path)
    out_img_paths = [out_path + '/' + s.replace('.dcm', '.pkl') for s in in_img_paths]
    in_img_paths = [in_path + '/' + s for s in in_img_paths]

    for in_path, out_path in zip(in_img_paths, out_img_paths):
        img, metadata = read_dicom(in_path)
        save_data(img, metadata, out_path)


def preprocess(in_data_path, out_data_path=None):
    dataset_name = in_data_path.split('/')[-1]
    out_data_path = in_data_path.replace(dataset_name, 'pkl_' + dataset_name)

    in_patient_dirs = sorted(os.listdir(in_data_path))
    out_patient_dirs = [out_data_path + '/' + s for s in in_patient_dirs]
    in_patient_dirs = [in_data_path + '/' + s for s in in_patient_dirs]

    for d_in, d_out in zip(in_patient_dirs, out_patient_dirs):
        print '\n******** %s *********' % d_in
        utils.automakedir(d_out)
        convert_patient_dicom2pkl(d_in, d_out)


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     sys.exit("Usage: dicom2npy.py <global_data_path>")
    # global_path = sys.argv[1]
    global_path = '/mnt/sda3/data/kaggle-lung/sample_images'
    preprocess(global_path)
