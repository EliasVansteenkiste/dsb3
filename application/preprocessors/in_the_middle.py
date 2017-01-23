from interfaces.preprocess import BasePreprocessor
from interfaces.data_loader import INPUT
from utils import put_in_the_middle
import numpy as np

class PutInTheMiddle(BasePreprocessor):

    def __init__(self, tag=None, output_shape=(256,256,256)):
        assert tag is not None
        self.output_shape = output_shape
        self.tag = tag

    def process(self, sample):
        for input_key, value in sample[INPUT].iteritems():
            if self.tag in input_key:
                sample[INPUT][input_key] = put_in_the_middle(target_tensor=np.zeros(shape=self.output_shape),
                                                             data_tensor = value
                                                             )
        return sample
