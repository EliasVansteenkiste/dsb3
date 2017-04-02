import utils
import utils_lung
import numpy as np

path1 = '/mnt/storage/metadata/dsb3/model-predictions/eavsteen/dsb_c3_s2_p8a1_ls_elias/'
path2 = '/mnt/storage/metadata/dsb3/model-predictions/ikorshun/dsb_c3_s1e_p8a1/'

id2c1 = utils_lung.get_candidates_paths(path1)
id2c2 = utils_lung.get_candidates_paths(path2)

for pid in id2c1.iterkeys():
    print pid
    d1 = utils.load_pkl(id2c1[pid])
    d2 = utils.load_pkl(id2c2[pid])
    assert np.all(d1 == d2)
