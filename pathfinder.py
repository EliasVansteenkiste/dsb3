import json
import utils
import os

if utils.hostname() == 'user':
    with open('SETTINGS_local.json') as data_file:
        paths = json.load(data_file)
else:
    with open('SETTINGS.json') as data_file:
        paths = json.load(data_file)

DATA_PATH = paths["DATA_PATH"]
utils.check_data_paths(DATA_PATH)

LUNA_DATA_PATH = paths["LUNA_DATA_PATH"]
utils.check_data_paths(LUNA_DATA_PATH)

LABELS_PATH = paths["LABELS_PATH"]
if not os.path.isfile(LABELS_PATH):
    raise ValueError('no file with train labels')

LUNA_LABELS_PATH = paths["LUNA_LABELS_PATH"]
if not os.path.isfile(LUNA_LABELS_PATH):
    raise ValueError('no file with luna annotations')

SAMPLE_SUBMISSION_PATH = paths["SAMPLE_SUBMISSION_PATH"]
if not os.path.isfile(SAMPLE_SUBMISSION_PATH):
    raise ValueError('no sample submission file')

VALIDATION_SPLIT_PATH = paths["VALIDATION_SPLIT_PATH"]
# if not os.path.isfile(VALIDATION_SPLIT_PATH):
#     raise ValueError('no validation file')

METADATA_PATH = paths["METADATA_PATH"]
