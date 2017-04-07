## TODO

in utils.py do:
no_name=True like
def get_dir_path(dir_name, root_dir, no_name=True):
...

when the stage 2 data we need to change settings.json:
 "DATA_PATH_2": "/data/dsb3/stage1/"
 "SAMPLE_SUBMISSION_PATH_2": "/mnt/storage/data/dsb3/stage1_sample_submission.csv"
 add test_stage2 pids to "VALIDATION_SPLIT_PATH": "/mnt/storage/data/dsb3/dsb_validation_split.pkl",
