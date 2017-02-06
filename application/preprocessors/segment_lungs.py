import numpy as np
import random
import math

from interfaces.preprocess import BasePreprocessor
from interfaces.data_loader import INPUT, OUTPUT

from scripts.elias.segmentation_kernel import segment_lung_from_ct_scan


class SegmentLungs(BasePreprocessor):
    def __init__(self, tags):
        self.tags = tags

    def process(self, sample):
        for tag in self.tags:
            if tag in sample[INPUT]:
                volume = sample[INPUT][tag]
                sample[INPUT][tag] = segment_lung_from_ct_scan(volume, plot=True, savefig=True)
            elif tag in sample[OUTPUT]:
                volume = sample[OUTPUT][tag]
                sample[OUTPUT][tag] = segment_lung_from_ct_scan(volume, plot=True, savefig=True)
