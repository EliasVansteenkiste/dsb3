import random
from interfaces.preprocess import BasePreprocessor

from utils.image_transform import resize_and_augment


DEFAULT_AUGMENTATION_PARAMETERS = {
    "zoom_x": [1, 1],  # factor
    "zoom_y": [1, 1],  # factor
    "rotate": [0, 0],  # degrees
    "skew_x": [0, 0],  # degrees
    "skew_y": [0, 0],  # degrees
    "translate_x": [0, 0],  # pixels
    "translate_y": [0, 0],  # pixels
    "change_brightness": [0, 0],
}

quasi_random_generator = None

def sample_augmentation_parameters(augmentation_params):
    augm = dict(augmentation_params)

    for key, value in augm.iteritems():
        if isinstance(augm[key],str):
            augm[key] = DEFAULT_AUGMENTATION_PARAMETERS[key]
    augm = dict(DEFAULT_AUGMENTATION_PARAMETERS, **augm)

    res = dict()

    for key, (a, b) in augm.iteritems():
        res[key] = a + random.random()*(b-a)

    for key, value in augmentation_params.iteritems():
        if isinstance(augmentation_params[key],str):
            res[key] = res[value]  # copy the contents of the other field!

    return res

class AugmentInput(BasePreprocessor):
    def __init__(self, output_shape=(56,56), **augmentation_params):
        self.augmentation_params = augmentation_params
        self.output_shape = output_shape

    def process(self, sample):
        augment_p = sample_augmentation_parameters(self.augmentation_params)
        from interfaces.data_loader import INPUT
        for input_key, value in sample[INPUT].iteritems():
            image = sample[INPUT][input_key]
            sample[INPUT][input_key] = resize_and_augment(image, output_shape=self.output_shape, augment=augment_p)
        return sample