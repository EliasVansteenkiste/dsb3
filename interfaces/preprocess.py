from collections import defaultdict
import itertools
import random

import numpy as np

from utils.image_transform import resize_and_augment


DEFAULT_AUGMENTATION_PARAMETERS = {
    "zoom_x":[1, 1],  # factor
    "zoom_y":[1, 1],  # factor
    "rotate":[0, 0],  # degrees
    "skew_x":[0, 0],  # degrees
    "skew_y":[0, 0],  # degrees
    "translate_x":[0, 0],  # pixels
    "translate_y":[0, 0],  # pixels
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


class BasePreprocessor(object):
    @property
    def extra_input_tags_required(self):
        return []

    @property
    def extra_output_tags_required(self):
        return []

    def train(self, data_iterator):
        pass

    def process(self, sample):
        pass

class NormalizeInput(BasePreprocessor):

    def __init__(self, num_samples = 100):
        self.std = dict()
        self.bias = dict()
        self.num_samples = num_samples

    def train(self, data_iterator):
        if len(self.std) != 0 and len(self.bias) != 0:
            return

        print "Training NormalizeInput preprocessor..."
        from interfaces.data_loader import INPUT

        d = defaultdict(list)
        for _, sample in itertools.izip(xrange(self.num_samples), data_iterator):
            for input_key, value in sample[INPUT].iteritems():
                d[input_key].append(value.flatten())

        for input_key, value in d.iteritems():
            d[input_key] = np.concatenate(d[input_key])
            self.std[input_key] = np.std(np.array(d[input_key]))
            self.bias[input_key] = np.mean(np.array(d[input_key]))

    def process(self, sample):
        from interfaces.data_loader import INPUT
        for input_key, value in sample[INPUT].iteritems():
            sample[INPUT][input_key] = (value-self.bias[input_key]) / self.std[input_key]
        return sample

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


class RescaleInput(BasePreprocessor):
    def __init__(self, input_scale=(0,255), output_scale=(0.0, 1.0)):
        self.input_scale = input_scale
        self.output_scale = output_scale

        self.coef = (input_scale[1]-input_scale[0])/(output_scale[1]-output_scale[0])
        self.bias = self.output_scale[0] - (self.input_scale[0] / self.coef)

    def process(self, sample):
        from interfaces.data_loader import INPUT
        for input_key, value in sample[INPUT].iteritems():
            image = sample[INPUT][input_key]
            sample[INPUT][input_key] = (image - self.bias) / self.coef
        return sample