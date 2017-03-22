from collections import defaultdict
import itertools
import random

import numpy as np


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
        #raise NotImplementedError("Don't use NormalizeInput. It does not reload parameters properly")
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

        print "std:", self.std
        print "bias:", self.bias

    def process(self, sample):
        from interfaces.data_loader import INPUT
        for input_key, value in sample[INPUT].iteritems():
            sample[INPUT][input_key] = (value-self.bias[input_key]) / self.std[input_key]


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


class ZMUV(BasePreprocessor):
    def __init__(self, tag, bias, std):
        self.tag = tag
        self.bias = bias
        self.std = std

    def process(self, sample):
        from interfaces.data_loader import INPUT, OUTPUT
        if self.tag in sample[INPUT]:
            image = sample[INPUT][self.tag]
            sample[INPUT][self.tag] = (image - self.bias) / self.std
        elif self.tag in sample[OUTPUT]:
            image = sample[OUTPUT][self.tag]
            sample[OUTPUT][self.tag] = (image - self.bias) / self.std