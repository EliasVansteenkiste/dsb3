import pathfinder
import glob
import random
import utils_lung
import utils

VALIDATION_SET_SIZE = 0.2


def read_split(path):
    d = utils.load_pkl(path)
    print d['valid']
    # print d['train']


def make_luna_validation_split():
    luna_path = pathfinder.LUNA_DATA_PATH
    file_list = sorted(glob.glob(luna_path + "/*.mhd"))
    random.seed(317070)
    all_pids = [utils_lung.luna_extract_pid(f) for f in file_list]
    validation_pids = random.sample(all_pids, int(VALIDATION_SET_SIZE * len(file_list)))
    train_pids = list(set(all_pids) - set(validation_pids))
    d = {}
    d['valid'] = validation_pids
    d['train'] = train_pids
    utils.save_pkl(d, pathfinder.LUNA_VALIDATION_SPLIT_PATH)


def make_kaggle_validation_split():
    pass


if __name__ == '__main__':
    make_luna_validation_split()
    read_split(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
