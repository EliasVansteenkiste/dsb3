import numpy as np
import random
import math

from interfaces.preprocess import BasePreprocessor
from interfaces.data_loader import INPUT, OUTPUT

from segmentation_kernel import segment_lung_from_ct_scan


class SegmentLungs(BasePreprocessor):
    def __init__(self, input_scale=(0,255), output_scale=(0.0, 1.0)):
        self.input_scale = input_scale
        self.output_scale = output_scale


    def process(self, sample):
        from interfaces.data_loader import INPUT
        for input_key, value in sample[INPUT].iteritems():
            image = sample[INPUT][input_key]
            print image.shape
            segmented_image = segment_lung_from_ct_scan(image, plot=True, savefig=True)
            print segmented_image.shape
            sample[INPUT][input_key] = segmented_image