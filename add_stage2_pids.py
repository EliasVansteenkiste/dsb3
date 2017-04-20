import pathfinder
import utils_lung
import utils

print pathfinder.SAMPLE_SUBMISSION_PATH
id2labels = utils_lung.read_sample_submission(pathfinder.SAMPLE_SUBMISSION_PATH)
d = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
d['test_stage2'] = id2labels.keys()
utils.save_pkl(d, pathfinder.VALIDATION_SPLIT_PATH)
