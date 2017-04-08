import json
import utils
import os
import utils_lung

if utils.hostname() == 'user':
    with open('SETTINGS_user.json') as data_file:
        paths = json.load(data_file)
else:
    with open('SETTINGS.json') as data_file:
        paths = json.load(data_file)

# kaggle data
STAGE = int(paths["STAGE"])

if STAGE == 1:
    METADATA_PATH = paths["METADATA_PATH_1"]

    DATA_PATH = paths["DATA_PATH_1"]
    utils.check_data_paths(DATA_PATH)

    SAMPLE_SUBMISSION_PATH = paths["SAMPLE_SUBMISSION_PATH_1"]
    if not os.path.isfile(SAMPLE_SUBMISSION_PATH):
        raise ValueError('no stage 1 sample submission file')

elif STAGE == 2:
    METADATA_PATH = paths["METADATA_PATH_2"]

    DATA_PATH = paths["DATA_PATH_2"]
    utils.check_data_paths(DATA_PATH)

    SAMPLE_SUBMISSION_PATH = paths["SAMPLE_SUBMISSION_PATH_2"]
    if not os.path.isfile(SAMPLE_SUBMISSION_PATH):
        raise ValueError('no stage 2 sample submission file')

LABELS_PATH = paths["LABELS_PATH"]
if not os.path.isfile(LABELS_PATH):
    raise ValueError('no file with train labels')


TEST_LABELS_PATH = paths["TEST_LABELS_PATH"]
if not os.path.isfile(TEST_LABELS_PATH):
    raise ValueError('no file with test labels')

VALIDATION_SPLIT_PATH = paths["VALIDATION_SPLIT_PATH"]
if not os.path.isfile(VALIDATION_SPLIT_PATH):
    raise ValueError('no VALIDATION_SPLIT_PATH file')

FINAL_SPLIT_PATH = paths["FINAL_SPLIT_PATH"]
if not os.path.isfile(FINAL_SPLIT_PATH):
    raise ValueError('no FINAL_SPLIT_PATH file')

# luna data
LUNA_DATA_PATH = paths["LUNA_DATA_PATH"]
utils.check_data_paths(LUNA_DATA_PATH)

LUNA_SEG_DATA_PATH = paths["LUNA_SEG_DATA_PATH"]
utils.check_data_paths(LUNA_SEG_DATA_PATH)

LUNA_LABELS_PATH = paths["LUNA_LABELS_PATH"]
if not os.path.isfile(LUNA_LABELS_PATH):
    raise ValueError('no file with luna annotations')

LUNA_CANDIDATES_PATH = paths["LUNA_CANDIDATES_PATH"]
if not os.path.isfile(LUNA_CANDIDATES_PATH):
    raise ValueError('no LUNA file candidates_V2.csv')

LUNA_VALIDATION_SPLIT_PATH = paths["LUNA_VALIDATION_SPLIT_PATH"]
if not os.path.isfile(LUNA_VALIDATION_SPLIT_PATH):
    raise ValueError('no LUNA validation split file')

LUNA_NODULE_ANNOTATIONS_PATH = paths["LUNA_NODULE_ANNOTATIONS"]
utils.check_data_paths(LUNA_NODULE_ANNOTATIONS_PATH)

LUNA_PROPERTIES_PATH = paths['LUNA_PROPERTIES_PATH']
if not os.path.isfile(LUNA_PROPERTIES_PATH):
    raise ValueError('no LUNA properties file')

VALIDATION_LB_MIXED_SPLIT_PATH = paths['VALIDATION_LB_MIXED_SPLIT_PATH']
if not os.path.isfile(VALIDATION_LB_MIXED_SPLIT_PATH):
    raise ValueError('no mixed validation and LB file')
