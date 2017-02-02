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
        input_tags_extra = [dsn+":metadata" for dsn in datasetnames]
        return input_tags_extra

    def process(self, sample):
        for tag in self.tags:
            metadatatag = tag.split(':')[0]+":metadata"
            assert metadatatag in sample[INPUT], "tag %s not found"%metadatatag
            metadata = sample[INPUT][metadatatag]

            if tag in sample[INPUT]:
                sample[INPUT][tag] = self.convert(sample[INPUT][tag], metadata)
            elif tag in sample[OUTPUT]:
                sample[INPUT][tag] = self.convert(sample[OUTPUT][tag], metadata)
            else:
                raise Exception("Did not find tag which I had to augment: %s"%tag)

    @staticmethod
    def convert(data, metadata):
        #sort the slices by image position
        image_positions = [i[2] for i in metadata["imagepositions"]]
        sort_ids = np.argsort(image_positions)
        data = data.T[sort_ids] #ZYX to XYZ and sort
        intercepts = metadata["rescaleintercept"][sort_ids]
        slopes = metadata["rescaleslope"][sort_ids]

        # Set outside-of-scan pixels to 0, The intercept is usually -1024, so air is approximately 0
        data[data == -2000] = 0

        # Convert to Hounsfield units (HU)
        if slopes[0] != 1:
            data = slopes[0] * data.astype(np.float64)
            data = data.astype(np.int16)
        data += np.int16(intercepts[0])
        return data