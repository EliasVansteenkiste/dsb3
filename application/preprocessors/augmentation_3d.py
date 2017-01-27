import numpy as np
import random
import math

from interfaces.preprocess import BasePreprocessor
from utils.transformation_3d import affine_transform, apply_affine_transform
from interfaces.data_loader import INPUT, OUTPUT


DEFAULT_AUGMENTATION_PARAMETERS = {
    "scale": [1, 1, 1],  # factor
    "uniform scale": 1, # factor, same scale in all directions
    "rotation": [0, 0, 0],  # degrees
    "shear": [0, 0, 0],  # degrees
    "translation": [0, 0, 0],  # mm
    "reflection": [0, 0, 0] #Bernoulli p
}

MIN_HU = -1000.


def log_uniform(max_val):
    return math.exp(uniform(math.log(max_val)))


def uniform(max_val):
    return max_val*(random.random()*2-1)


def bernoulli(p): return random.random() < p  #range [0.0, 1.0)


def augment_3d(volume, pixel_spacing, output_shape, norm_patch_shape, augment_p, interp_order=1):
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
    augments = affine_transform(**augment_p)
    patchscale = affine_transform(scale=patch_shape / norm_shape)
    unshift_center = affine_transform(translation=output_shape / 2. - 0.5)

    matrix = shift_center.dot(normscale).dot(augments).dot(patchscale).dot(unshift_center)

    output = apply_affine_transform(volume, matrix,
                                    order=interp_order,
                                    output_shape=output_shape.astype(np.int),
                                    cval=MIN_HU)
    return output


def sample_augmentation_parameters(augm):
    new_augm = dict(DEFAULT_AUGMENTATION_PARAMETERS)
    if "scale" in augm:
        new_augm["scale"] = [log_uniform(v) for v in augm["scale"]]
    if "uniform scale" in augm:
        uscale = log_uniform(augm["uniform scale"])
        new_augm["scale"] = [v*uscale for v in new_augm["scale"]]
    if "rotation" in augm:
        new_augm["rotation"] = [uniform(v) for v in augm["rotation"]]
    if "shear" in augm:
        new_augm["shear"] = [uniform(v) for v in augm["shear"]]
    if "translation" in augm:
        new_augm["translation"] = [uniform(v) for v in augm["translation"]]
    if "reflection" in augm:
        new_augm["reflection"] = [bernoulli(v) for v in augm["reflection"]]
    return new_augm


class Augment3D(BasePreprocessor):
    def __init__(self, tags, output_shape, norm_patch_shape,
                 augmentation_params=DEFAULT_AUGMENTATION_PARAMETERS,
                 interp_order=1):
        """
        :param output_shape: the output shape in shape of the array
        :param norm_patch_shape: the output shape in mm's
        :param augmentation_params: the parameters used for sampling the augmentation.
        :return:
        """
        self.augmentation_params = augmentation_params
        self.output_shape = output_shape
        self.norm_patch_shape = norm_patch_shape
        self.tags = tags
        self.interp_order = interp_order

    @property
    def extra_input_tags_required(self):
        datasetnames = set()
        for tag in self.tags:datasetnames.add(tag.split(':')[0])
        input_tags_extra = [dsn+":pixelspacing" for dsn in datasetnames]
        return input_tags_extra


    def process(self, sample):
        augment_p = sample_augmentation_parameters(self.augmentation_params)

        for tag in self.tags:
            pixelspacingtag = tag.split(':')[0]+":pixelspacing"
            assert pixelspacingtag in sample[INPUT], "tag %s not found"%pixelspacingtag
            spacing = sample[INPUT][pixelspacingtag]

            if tag in sample[INPUT]:
                volume = sample[INPUT][tag]
                sample[INPUT][tag] = augment_3d(
                    volume=volume,
                    pixel_spacing=spacing,
                    output_shape=self.output_shape,
                    norm_patch_shape=self.norm_patch_shape,
                    augment_p=augment_p,
                    interp_order = self.interp_order
                )
            elif tag in sample[OUTPUT]:
                volume = sample[OUTPUT][tag]
                sample[OUTPUT][tag] = augment_3d(
                    volume=volume,
                    pixel_spacing=spacing,
                    output_shape=self.output_shape,
                    norm_patch_shape=self.norm_patch_shape,
                    augment_p=augment_p,
                    interp_order=self.interp_order
                )
#            else:
#		print sample[INPUT].keys()
 #               raise Exception("Did not find tag which I had to augment: %s"%tag)


def test_augmentation():
    import bcolz, os, gzip, cPickle
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


        sample = {INPUT:{
            "bcolzall:volume":vol,
            "bcolzall:pixelspacing":spacings[patient]
        }}

        prep = Augment3D(
            tags=["bcolzall:volume"],
            output_shape = (320/2, 340/2, 340/2),
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
        prep.process(sample)
        print "prep", time() - t0

        vol_aug = sample[INPUT]["bcolzall:volume"]

        import utils.plt
        utils.plt.show_compare(vol, vol_aug)


if __name__ == '__main__':
    test_augmentation()
