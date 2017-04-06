import utils
import utils_lung
import numpy as np

path1 = '/mnt/storage/metadata/dsb3_stage2/model-predictions/dsb_c3_s5_p8a1/'
path2 = '/mnt/storage/metadata/dsb3/model-predictions/ikorshun/dsb_c3_s5_p8a1/'

id2c1 = utils_lung.get_candidates_paths(path1)
print id2c1
id2c2 = utils_lung.get_candidates_paths(path2)
# print id2c2

for pid in id2c1.iterkeys():
    print pid
    d1 = utils.load_pkl(id2c1[pid])
    d2 = utils.load_pkl(id2c2[pid])

    assert np.all(d1 == d2)