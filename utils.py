import cPickle
import platform
import pwd
import subprocess
import time
import numpy as np
import glob
import os

maxfloat = np.finfo(np.float32).max


def auto_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print 'Created dir', path


def find_model_metadata(metadata_dir, config_name):
    metadata_paths = glob.glob(metadata_dir + '/%s-*' % config_name)
    if not metadata_paths:
        raise ValueError('No metadata files for config %s' % config_name)
    elif len(metadata_paths) > 1:
        raise ValueError('Multiple metadata files for config %s' % config_name)
    print 'Loaded model from', metadata_paths[0]
    return metadata_paths[0]


def get_train_valid_split(train_data_path):
    filename = 'valid_split.pkl'
    # if not os.path.isfile(filename):
    #     print 'Making validation split'
    #     create_validation_split.save_train_validation_ids(filename, train_data_path)
    return load_pkl(filename)


def check_data_paths(data_path):
    if not os.path.isdir(data_path):
        raise ValueError('wrong path to DICOM data')


def get_dir_path(dir_name, root_dir):
    username = pwd.getpwuid(os.getuid())[0]
    dir_path = root_dir + '/' + dir_name + '/%s' % username
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def hostname():
    return platform.node()


def generate_expid(arch_name):
    return "%s-%s" % (arch_name, timestamp())


def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    except:
        return 0


def save_pkl(obj, path, protocol=cPickle.HIGHEST_PROTOCOL):
    with open(path, 'w') as f:
        cPickle.dump(obj, f, protocol=protocol)


def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
    return obj


def copy(from_folder, to_folder):
    command = "cp -r %s %s/." % (from_folder, to_folder)
    print command
    os.system(command)


def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]

    return current_lr
