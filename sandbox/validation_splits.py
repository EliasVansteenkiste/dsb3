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
    all_pids = [utils_lung.extract_pid_filename(f) for f in file_list]
    validation_pids = random.sample(all_pids, int(VALIDATION_SET_SIZE * len(file_list)))
    train_pids = list(set(all_pids) - set(validation_pids))
    d = {}
    d['valid'] = validation_pids
    d['train'] = train_pids
    utils.save_pkl(d, pathfinder.LUNA_VALIDATION_SPLIT_PATH)


def make_kaggle_validation_split():
    pass


def make_LB_based_validation():
    train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
    train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids[
        'test']


    random.seed(317070)
    half_valid_pids = random.sample(valid_pids, int(0.5 * len(valid_pids)))
    half_test_pids = random.sample(test_pids, int(0.5 * len(test_pids)))

    new_valid = half_valid_pids + half_test_pids
    new_test = list(set(valid_pids+test_pids) - set(new_valid))

    d = {}
    d['test'] = new_test
    d['validation'] = new_valid
    d['training'] = train_pids
    utils.save_pkl(d, pathfinder.VALIDATION_LB_MIXED_SPLIT_PATH)



if __name__ == '__main__':
    make_LB_based_validation()

