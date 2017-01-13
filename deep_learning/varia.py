import lasagne
import theano.tensor as T
from theano_utils import theano_printer
import numpy as np

class SelectLayer(lasagne.layers.MergeLayer):
    def __init__(self, layers, select_layer, *args, **kwargs):
        super(SelectLayer, self).__init__(layers+[select_layer], *args, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):

        all_output = T.concatenate([i.reshape(shape=(1,-1,np.prod(self.output_shape[1:]))) for i in inputs[:-1]], axis=0)
        idx = inputs[-1].astype('int64')-1
        selected = all_output[idx,T.arange(idx.shape[0])]
        selected = selected.reshape(shape = [-1 if i is None else i for i in self.output_shape])
        return selected

class PoolOverTimeLayer(lasagne.layers.GlobalPoolLayer):

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]

    def get_output_for(self, input, **kwargs):
        return self.pool_function(input, axis=len(self.output_shape))
