import json
import utils
import os
import utils_lung


with open('SETTINGS.json') as data_file:
    paths = json.load(data_file)


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
# if not os.path.isfile(VALIDATION_SPLIT_PATH):
#     raise ValueError('no validation file')

# luna data
LUNA_DATA_PATH = paths["LUNA_DATA_PATH"]
utils.check_data_paths(LUNA_DATA_PATH)

LUNA_LABELS_PATH = paths["LUNA_LABELS_PATH"]
if not os.path.isfile(LUNA_LABELS_PATH):
    raise ValueError('no file with luna annotations')

LUNA_CANDIDATES_PATH = paths["LUNA_CANDIDATES_PATH"]
if not os.path.isfile(LUNA_CANDIDATES_PATH):
    raise ValueError('no LUNA file candidates_V2.csv')

LUNA_VALIDATION_SPLIT_PATH = paths["LUNA_VALIDATION_SPLIT_PATH"]
if not os.path.isfile(LUNA_VALIDATION_SPLIT_PATH):
    raise ValueError('no LUNA validation split file')


LUNA_LUNG_SEG_PATH = paths["LUNA_LUNG_SEG_PATH"]

AAPM_DATA_PATH = paths["AAPM_DATA_PATH"]


AAPM_EXTENDED_DATA_PATH = paths["AAPM_EXTENDED_DATA_PATH"]


AAPM_LABELS_PATH = paths["AAPM_LABELS_PATH"]
print "lung seg path: {}".format(LUNA_LUNG_SEG_PATH)
if not os.path.isfile(AAPM_LABELS_PATH):
    raise ValueError('no AAPM labels csv found!')

AAPM_EXTENDED_LABELS_PATH = paths["AAPM_EXTENDED_LABELS_PATH"]
print "lung seg path: {}".format(LUNA_LUNG_SEG_PATH)
if not os.path.isfile(AAPM_EXTENDED_LABELS_PATH):
    raise ValueError('no extended AAPM labels csv found!')



AAPM_CANDIDATES_PATH = paths["AAPM_CANDIDATES_PATH"]
print "lung seg path: {}".format(AAPM_CANDIDATES_PATH)
if not os.path.isfile(AAPM_CANDIDATES_PATH):
    raise ValueError('no AAPM candidates csv found!')

LUNA_PROPERTIES_PATH = paths['LUNA_PROPERTIES_PATH']
if not os.path.isfile(LUNA_PROPERTIES_PATH):
    raise ValueError('no LUNA properties file')

VALIDATION_LB_MIXED_SPLIT_PATH = paths['VALIDATION_LB_MIXED_SPLIT_PATH']
if not os.path.isfile(VALIDATION_LB_MIXED_SPLIT_PATH):
    raise ValueError('no mixed validation and LB file')

FINAL_SPLIT_PATH = paths['FINAL_SPLIT_PATH']
if not os.path.isfile(FINAL_SPLIT_PATH):
    raise ValueError('no mixed validation and LB file')
