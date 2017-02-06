import numpy as np

from interfaces.preprocess import BasePreprocessor
from interfaces.data_loader import INPUT, OUTPUT


# Hounsfield Unit
class Stage1ToHU(BasePreprocessor):
    def __init__(self, tags):
        self.tags = tags

    @property
    def extra_input_tags_required(self):
        datasetnames = set()
        for tag in self.tags:datasetnames.add(tag.split(':')[0])
        input_tags_extra = [dsn+":slope" for dsn in datasetnames]
        input_tags_extra += [dsn+":intercept" for dsn in datasetnames]
        return input_tags_extra

    def process(self, sample):
        for tag in self.tags:
            slopetag = tag.split(':')[0]+":slope"
            assert slopetag in sample[INPUT], "tag %s not found"%slopetag
            slope = sample[INPUT][slopetag]

            intercepttag = tag.split(':')[0] + ":intercept"
            assert intercepttag in sample[INPUT], "tag %s not found" % intercepttag
            intercept = sample[INPUT][intercepttag]

            if tag in sample[INPUT]:
                sample[INPUT][tag] = self.convert(sample[INPUT][tag], slope, intercept)
            elif tag in sample[OUTPUT]:
                sample[INPUT][tag] = self.convert(sample[OUTPUT][tag], slope, intercept)

    @staticmethod
    def convert(data, slope, intercept):
        # Set outside-of-scan pixels to 0, The intercept is usually -1024, so air is approximately 0
        data[data == -2000] = 0

        # Convert to Hounsfield units (HU)
        if slope != 1:
            data = slope * data.astype(np.float64)
            data = data.astype(np.int16)
        data += np.int16(intercept)
        return data