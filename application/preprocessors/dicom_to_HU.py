import numpy as np

from interfaces.preprocess import BasePreprocessor
from interfaces.data_loader import INPUT, OUTPUT

def checkEqual(lst):
   return (lst[1:] == lst[:-1]).all()

def checkAbsSmall(lst, tolerance):
    return (abs(lst[1:]-lst[:-1])<tolerance).all()

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

        data = np.asarray([s["PixelData"] for s in slices], dtype=np.int16)
        pixelspacings = np.asarray([s["PixelSpacing"] for s in slices], dtype=np.float32)
        
        # Set outside-of-scan pixels to 0, The intercept is usually -1024, so air is approximately 0

        # Convert to Hounsfield units (HU)
        data = data.astype(np.float32)
        slopes = np.asarray([s["RescaleSlope"] for s in slices], dtype=np.float32)
        intercepts = np.asarray([s["RescaleIntercept"] for s in slices], dtype=np.float32)
        if any(slopes != 1.): 
            data = slopes[:,None,None] * data
        data += intercepts[:,None,None]
        data[data < -1000] = -1000
        #up until here multiple scan trials were not important


        #Check if there are multiple scan trials, let's look at the z-component
        image_positions = np.asarray([s["ImagePositionPatient"][2] for s in slices])
        sorted_image_positions = np.sort(image_positions)
        zdelta = sorted_image_positions[1::] - sorted_image_positions[:-1:]
        if any(zdelta < 0.001):
            print "Warning found a scan with multiple scan trials"
            print 'image_positions'
            print image_positions
            print 'sorted_image_positions'
            print sorted_image_positions
            print 'InstanceNumber'
            instance_numbers = np.asarray([s["InstanceNumber"] for s in slices])
            print instance_numbers
            idcs = np.greater(instance_numbers,len(instance_numbers)/2)
            print 'idcs', idcs
            print 'idcs.dtype', idcs.dtype

            print 'we assume there is only two scan trials, please check this in the printed positions'
            assert(data.shape[0]%2==0)
            print 'data.shape', data.shape
            data = data[idcs]
            print 'data.shape', data.shape
            print 'len(image_positions)', len(image_positions)
            image_positions = image_positions[idcs]
            print 'len(image_positions)', len(image_positions)
            sorted_image_positions = np.sort(image_positions)
            pixelspacings = pixelspacings[idcs]
            zdelta = sorted_image_positions[1::] - sorted_image_positions[:-1:]


        sort_ids = np.argsort(image_positions)


        data = data[sort_ids].T # ZYX to XYZ and sort
        
        #check if spacings are homogenous
        assert(checkAbsSmall(zdelta,tolerance=0.003)==True), zdelta
        assert(checkEqual(pixelspacings)==True), pixelspacings

        spacing = [pixelspacings[0,1], pixelspacings[0,0], zdelta[0]]

        return {tag: data, tag.split(":")[0]+":pixelspacing": spacing}
