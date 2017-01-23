import numpy as np
import random
import math

from interfaces.preprocess import BasePreprocessor
from utils.transformation_3d import affine_transform, apply_affine_transform
from interfaces.data_loader import INPUT, OUTPUT


DEFAULT_AUGMENTATION_PARAMETERS = {
    "scale": [1, 1, 1],  # factor
    "rotation": [0, 0, 0],  # degrees
    "shear": [0, 0, 0],  # degrees
    "translation": [0, 0, 0],  # pixels
    "reflection": [0, 0, 0] #Bernoulli p
}


def lio_augment(volume, pixel_spacing, output_shape, norm_patch_size, augment_p):
    input_shape = np.asarray(volume.shape, np.float)
    pixel_spacing = np.asarray(pixel_spacing, np.float)
    output_shape = np.asarray(output_shape, np.float)
    norm_patch_size = np.asarray(norm_patch_size, np.float)

    norm_shape = input_shape * pixel_spacing
    # this will stretch in some dimensions, but the stretch is consistent across samples
    patch_shape = norm_shape * output_shape / norm_patch_size
    # else, use this: patch_shape = norm_shape * np.min(output_shape / norm_patch_size)

    shift_center = affine_transform(translation=-input_shape / 2. - 0.5)
    normscale = affine_transform(scale=norm_shape / input_shape)
    augment = affine_transform(**augment_p)
    patchscale = affine_transform(scale=patch_shape / norm_shape)
    unshift_center = affine_transform(translation=output_shape / 2. - 0.5)

    matrix = shift_center.dot(normscale).dot(augment).dot(patchscale).dot(unshift_center)

    output = apply_affine_transform(volume, matrix, order=1, output_shape=output_shape.astype("int"))
    return output


def log_uniform(max_val):
    return math.exp(uniform(math.log(max_val)))


def uniform(max_val):
    return max_val*(random.random()*2-1)


def bernoulli(p):
    return p < random.random() #range [0.0, 1.0)


def sample_augmentation_parameters(augm):
    augm["scale"] = [log_uniform(v) for v in augm["scale"]]
    augm["rotation"] = [uniform(v) for v in augm["rotation"]]
    augm["shear"] = [uniform(v) for v in augm["shear"]]
    augm["translation"] = [uniform(v) for v in augm["translation"]]
    augm["reflection"] = [bernoulli(v) for v in augm["reflection"]]
    return augm


class LioAugment(BasePreprocessor):
    def __init__(self, tags, output_shape, norm_patch_size, augmentation_params=DEFAULT_AUGMENTATION_PARAMETERS):
        """
        :param output_shape: the output shape in shape of the array
        :param norm_patch_size: the output shape in mm's
        :param augmentation_params: the parameters used for sampling the augmentation.
        :return:
        """
        self.augmentation_params = augmentation_params
        self.output_shape = output_shape
        self.norm_patch_size = norm_patch_size
        self.tags = tags

    @property
    def extra_input_tags_required(self):
        """
        We need some extra parameters to be loaded!
        :return:
        """
        datasetnames = set()
        for tag in self.tags:
            datasetnames.add(tag.split(':')[0])

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
                sample[INPUT][tag] = lio_augment(
                    volume=volume,
                    pixel_spacing=spacing,
                    output_shape=self.output_shape,
                    norm_patch_size=self.norm_patch_size,
                    augment_p=augment_p
                )
            elif tag in sample[OUTPUT]:
                volume = sample[OUTPUT][tag]
                sample[OUTPUT][tag] = lio_augment(
                    volume=volume,
                    pixel_spacing=spacing,
                    output_shape=self.output_shape,
                    norm_patch_size=self.norm_patch_size,
                    augment_p=augment_p
                )
            else:
                pass
                # raise Exception("Did not find tag which I had to augment: %s"%tag)