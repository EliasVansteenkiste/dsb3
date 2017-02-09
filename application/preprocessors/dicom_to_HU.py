import numpy as np

from interfaces.preprocess import BasePreprocessor
from interfaces.data_loader import INPUT, OUTPUT


# Hounsfield Unit
class DicomToHU(BasePreprocessor):
    def __init__(self, tags): self.tags = tags

    def process(self, sample):
        for tag in self.tags:
            if tag in sample[INPUT]:
                sample[INPUT].update(self.convert(sample[INPUT][tag], tag))

    @property
    def extra_input_tags_required(self):
        datasetnames = set()
        for tag in self.tags:datasetnames.add(tag.split(':')[0])
        input_tags_extra = [dsn+":3d" for dsn in datasetnames]
        return input_tags_extra

    @staticmethod
    def convert(slices, tag):
        image_positions = [s["ImagePositionPatient"][2] for s in slices]
        sort_ids = np.argsort(image_positions)
        # print sort_ids.shape

        data = np.asarray([s["PixelData"] for s in slices], dtype=np.int16)
        data = data[sort_ids].T # ZYX to XYZ and sort

        # Set outside-of-scan pixels to 0, The intercept is usually -1024, so air is approximately 0
        data[data == -2000] = 0

        first_slice = slices[sort_ids[0]]
        slope = first_slice["RescaleSlope"]
        intercept = first_slice["RescaleIntercept"]
        # Convert to Hounsfield units (HU)
        data = data.astype(np.float32)
        if slope != 1: data = np.float32(slope) * data
        data += np.float32(intercept)

        image_positions.sort()
        spacing = first_slice["PixelSpacing"][::-1] + [image_positions[1] - image_positions[0]]

        # patients with multiple scan tries
        i = 0
        while spacing[2] < 0.001:
            spacing[2] = image_positions[2+i] - image_positions[0]
            i+=1

        return {tag: data,
                tag.split(":")[0]+":pixelspacing": spacing}
