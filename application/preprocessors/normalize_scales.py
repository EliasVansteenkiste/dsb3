from interfaces.preprocess import BasePreprocessor
from interfaces.data_loader import INPUT, OUTPUT


MAX_HU = 400.
MIN_HU = -1000.
PIXEL_MEAN = 0.25
NORMSCALE = 1./(MAX_HU - MIN_HU)
NORMOFFSET = - MIN_HU*NORMSCALE - PIXEL_MEAN


class DefaultNormalizer(BasePreprocessor):
    def __init__(self, tags, scale=NORMSCALE, offset=NORMOFFSET):
        self.tags = tags
        self.scale = scale
        self.offset = offset

    def process(self, sample):
        for tag in self.tags:
            if tag in sample[INPUT]:
                volume = sample[INPUT][tag]
                sample[INPUT][tag] = volume*self.scale+self.offset
            elif tag in sample[OUTPUT]:
                volume = sample[OUTPUT][tag]
                sample[OUTPUT][tag] = volume*self.scale+self.offset
            else:
                raise Exception("Did not find tag which I had to augment: %s"%tag)
