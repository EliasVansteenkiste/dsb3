import numpy as np
import random
import math

from interfaces.preprocess import BasePreprocessor
from utils.transformation_3d import affine_transform, apply_affine_transform
from interfaces.data_loader import INPUT


DEFAULT_AUGMENTATION_PARAMETERS = {
    "scale": [1, 1, 1],  # factor
    "rotation": [0, 0, 0],  # degrees
    "shear": [0, 0, 0],  # degrees
    "translation": [0, 0, 0],  # mm
    "reflection": [0, 0, 0] #Bernoulli p
}


def log_uniform(max_val):
    return math.exp(uniform(math.log(max_val)))


def uniform(max_val):
    return max_val*(random.random()*2-1)


def bernoulli(p): return random.random() < p  #range [0.0, 1.0)


MAX_HU = 400.
MIN_HU = -1000.
PIXEL_MEAN = 0.25
NORMSCALE = 1./(MAX_HU - MIN_HU)
NORMOFFSET = - MIN_HU*NORMSCALE - PIXEL_MEAN
def normalize_and_center(x): return x*NORMSCALE + NORMOFFSET


def lio_augment(volume, pixel_spacing, output_shape, norm_patch_shape, augment_p, interp_order=1):
    input_shape = np.asarray(volume.shape, np.float)
    pixel_spacing = np.asarray(pixel_spacing, np.float)
    output_shape = np.asarray(output_shape, np.float)
    norm_patch_shape = np.asarray(norm_patch_shape, np.float)

    norm_shape = input_shape * pixel_spacing
    # this will stretch in some dimensions, but the stretch is consistent across samples
    patch_shape = norm_shape * output_shape / norm_patch_shape
    # else, use this: patch_shape = norm_shape * np.min(output_shape / norm_patch_shape)

    shift_center = affine_transform(translation=-input_shape / 2. - 0.5)
    normscale = affine_transform(scale=norm_shape / input_shape)
    augment = affine_transform(**augment_p)
    patchscale = affine_transform(scale=patch_shape / norm_shape)
    unshift_center = affine_transform(translation=output_shape / 2. - 0.5)

    matrix = shift_center.dot(normscale).dot(augment).dot(patchscale).dot(unshift_center)

    output = apply_affine_transform(volume, matrix,
                                    order=interp_order,
                                    output_shape=output_shape.astype("int"),
                                    cval=MIN_HU)
    return output


def sample_augmentation_parameters(augm):
    augm["scale"] = [log_uniform(v) for v in augm["scale"]]
    augm["rotation"] = [uniform(v) for v in augm["rotation"]]
    augm["shear"] = [uniform(v) for v in augm["shear"]]
    augm["translation"] = [uniform(v) for v in augm["translation"]]
    augm["reflection"] = [bernoulli(v) for v in augm["reflection"]]
    return augm


class LioAugment(BasePreprocessor):
    def __init__(self, output_shape, norm_patch_shape, augmentation_params, interp_order=1):
        self.augmentation_params = augmentation_params
        self.output_shape = output_shape
        self.norm_patch_shape = norm_patch_shape
        self.interp_order = interp_order

    def process(self, sample):
        augment_p = sample_augmentation_parameters(self.augmentation_params)
        spacing = sample[INPUT]["spacing"]
        volume = sample[INPUT]["volume"]
        volume = lio_augment(
            volume=volume,
            pixel_spacing=spacing,
            output_shape=self.output_shape,
            norm_patch_shape=self.norm_patch_shape,
            augment_p=augment_p,
            interp_order=self.interp_order
        )
        volume = normalize_and_center(volume)
        sample[INPUT]["volume"] = volume
        return sample


def test_augmentation():
    import bcolz, os, gzip, cPickle
    import matplotlib.pyplot as plt
    from time import time
    INPUT_FOLDER = "/home/lio/data/dsb3/stage1+luna_bcolz/"
    SPACINGS_FILE = "/home/lio/data/dsb3/spacings.pkl.gz"

    with gzip.open(SPACINGS_FILE) as f:
        spacings = cPickle.load(f)

    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    for patient in patients:
        t0 = time()
        vol = bcolz.open(INPUT_FOLDER+patient)
        print "loading", time()-t0


        sample = {"input":{
            "volume":vol,
            "spacing":spacings[patient]
        }}

        prep = LioAugment(output_shape = (320/2, 340/2, 340/2),
                          norm_patch_shape=(320, 340, 340),
                          augmentation_params={
                            "scale": [1.05, 1.05, 1.05],  # factor
                            "rotation": [5, 5, 5],  # degrees
                            "shear": [2, 2, 2],  # degrees
                            "translation": [20, 20, 20],  # m
                            "reflection": [0, 0, 0] #Bernoulli p
                          },
                          interp_order=1)

        t0 = time()
        sample = prep.process(sample)
        print "prep", time() - t0

        vol_aug = sample["input"]["volume"]
        plt.close('all')
        fig, ax = plt.subplots(2, 3, figsize=(14,8))
        ax[0, 0].imshow(vol[vol.shape[0] // 2] , cmap="gray")
        ax[0, 1].imshow(vol[:, vol.shape[1] // 2] , cmap="gray")
        ax[0, 2].imshow(vol[:, :, vol.shape[2] // 2], cmap="gray")
        ax[1, 0].imshow(vol_aug[vol_aug.shape[0] // 2], cmap="gray")
        ax[1, 1].imshow(vol_aug[:, vol_aug.shape[1] // 2], cmap="gray")
        ax[1, 2].imshow(vol_aug[:, :, vol_aug.shape[2] // 2], cmap="gray")
        plt.show()


if __name__ == '__main__':
    test_augmentation()
