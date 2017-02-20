import pickle
import numpy as np

import utils.plt

p = pickle.load(open("/home/lio/data/dsb3/preds.pkl", "rb"))

     #.reshape((420,420,420))

patch_count = np.asarray([3,3,3], np.int)
stride = np.asarray([140, 140, 140], np.float)
shape = (patch_count*stride).astype("int")
norm_shape = np.asarray([ 330.,330.,349.19999075])


def glue(p):
    preds = []
    for x in range(patch_count[0]):
        preds_y = []
        for y in range(patch_count[1]):
            ofs = y*patch_count[2]+x*patch_count[2]*patch_count[1]
            preds_z = np.concatenate(p[ofs:ofs+patch_count[2]], axis=2)
            preds_y.append(preds_z)
        preds_y = np.concatenate(preds_y, axis=1)
        preds.append(preds_y)

    preds = np.concatenate(preds, axis=0)
    preds = preds[:int(round(norm_shape[0])), :int(round(norm_shape[1])), :int(round(norm_shape[2]))]
    return preds


preds = glue(p)

p = pickle.load(open("/home/lio/data/dsb3/patches.pkl", "rb"))
for i, patch in enumerate(p):
    p[i] = patch[10:-10, 10:-10, 10:-10]
patches = glue(p)

utils.plt.cross_sections([preds, patches], show=True)