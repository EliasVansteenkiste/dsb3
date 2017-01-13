"""
Library providing custom Lasagne layers for convolutions and pooling.

Copy-pasta from kaggle-heart
This is version 1.1
"""

import lasagne
import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.cuda import dnn
from lasagne.utils import as_tuple


if not theano.sandbox.cuda.cuda_enabled:
    raise ImportError(
            "requires GPU support -- see http://lasagne.readthedocs.org/en/"
            "latest/user/installation.html#gpu-support")  # pragma: no cover
elif not dnn.dnn_available():
    raise ImportError(
            "cuDNN not available: %s\nSee http://lasagne.readthedocs.org/en/"
            "latest/user/installation.html#cudnn" %
            dnn.dnn_available.msg)  # pragma: no cover


def replace_None(iterator):
    """
    Replace None by -1 in iterator, useful for reshapes in lasagna
    :param iterator:
    :return:
    """
    return [i if i is not None else -1 for i in iterator]

class ConvolutionLayer(lasagne.layers.Layer):
    def __init__(self,
                 incoming,
                 filter_mask_size = (3, 3),
                 filter_shape=(1, ),
                 convolution_axes=(2, 3),
                 channel_axes=(1, ),
                 stride=1,
                 width=1,
                 pad='same',
                 untie_biases=False,
                 W=lasagne.init.Orthogonal("relu"),
                 b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 flip_filters=False,
                 **kwargs):

        assert len(filter_shape) == len(channel_axes)
        assert len(filter_mask_size) == len(convolution_axes)

        super(ConvolutionLayer, self).__init__(incoming, **kwargs)

        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        n = len(convolution_axes)

        self.channel_axes = channel_axes
        self.convolution_axes = convolution_axes
        self.filter_shape = filter_shape
        self.n = n
        self.num_filters = np.prod(filter_shape)
        self.filter_mask_size = filter_mask_size
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.width = as_tuple(width, n, int)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_mask_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        if W is None:
            self.W = None
        else:
            self.W = self.add_param(W, self.get_W_shape(), name="W")

        if b is None:
            self.b = None
        else:
            if self.untie_biases is False:
                biases_shape = self.filter_shape
            else:
                biases_shape = self.filter_shape + tuple([self.output_shape[i] for i in self.convolution_axes])
            self.b = self.add_param(b, biases_shape, name="b", regularizable=False)

        assert all([self.input_shape[self.convolution_axes[i]] % (self.width[i]*self.stride[i]) == 0 for i in xrange(len(self.filter_mask_size))])


    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = sum([self.input_shape[i] for i in self.channel_axes])
        return (self.num_filters, num_input_channels) + self.filter_mask_size


    def get_convolutional_output_shape(self, input_shape):
        shape = []
        # axis channel shrinks
        for i in xrange(len(self.convolution_axes)):
            if isinstance(self.pad, tuple):
                shape.append(lasagne.layers.conv.conv_output_length(input_shape[self.convolution_axes[i]], self.filter_mask_size[i], self.stride[i], self.pad[i]))
            else:
                shape.append(lasagne.layers.conv.conv_output_length(input_shape[self.convolution_axes[i]], self.filter_mask_size[i], self.stride[i], self.pad))
        return shape


    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        # axis channel shrinks
        for i in xrange(len(self.convolution_axes)):
            if isinstance(self.pad, tuple):
                shape[self.convolution_axes[i]] = lasagne.layers.conv.conv_output_length(shape[self.convolution_axes[i]], self.filter_mask_size[i], self.stride[i], self.pad[i])
            else:
                shape[self.convolution_axes[i]] = lasagne.layers.conv.conv_output_length(shape[self.convolution_axes[i]], self.filter_mask_size[i], self.stride[i], self.pad)
        # filter channel changes
        for channel, filters in zip(self.channel_axes, self.filter_shape):
            shape[channel] = filters
        return tuple(shape)


    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            shuffle = ['x']*len(self.input_shape)
            for i,channel in enumerate(self.channel_axes + self.convolution_axes):
                shuffle[channel] = i
            activation = conved + self.b.dimshuffle(tuple(shuffle))
        else:
            shuffle = ['x']*len(self.input_shape)
            for i,channel in enumerate(self.channel_axes):
                shuffle[channel] = i
            activation = conved + self.b.dimshuffle(tuple(shuffle))
        return self.nonlinearity(activation)

    @property
    def filter_tensor(self):
        return self.W

    def convolve(self, input, **kwargs):
        """
            Reshape and move the desired channels to the back, and the other channels to the front.
        """
        dim_order = range(len(self.input_shape))
        for channel in self.channel_axes:
            dim_order.remove(channel)
        for axis in self.convolution_axes:
            dim_order.remove(axis)

        for channel in self.channel_axes:
            dim_order.append(channel)
        for axis in self.convolution_axes:
            dim_order.append(axis)

        if dim_order!=range(len(self.input_shape)):
            input = input.dimshuffle(*dim_order)

        # flatten other axes, flatten channel axes
        standard_input_shape = [-1, np.prod([self.input_shape[channel] for channel in self.channel_axes])] + [self.input_shape[axis] for axis in self.convolution_axes]
        input = input.reshape(standard_input_shape)

        target_shape, dim_shuffle_width = None,None
        if any([w!=1 for w in self.width]):
            target_shape = standard_input_shape[:2]
            dim_shuffle_width = [0]
            for i, width in enumerate(self.width):
                target_shape.append(standard_input_shape[2+i] // width)
                target_shape.append(width)
                dim_shuffle_width += [3+i*2]

            dim_shuffle_width += [1]
            for i in xrange(len(self.convolution_axes)):
                dim_shuffle_width += [2+i*2]

            input = input.reshape(target_shape).dimshuffle(*dim_shuffle_width)

            target_shape = [-1,  target_shape[1]]
            for i, width in enumerate(self.width):
                target_shape.append(standard_input_shape[2+i] // width)
            input = input.reshape(target_shape)

        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_mask_size)


        if len(self.convolution_axes)==0:
            conved_result = T.dot(input, self.filter_tensor.dimshuffle(1,0))
        elif len(self.convolution_axes)==1:
            """ perform convolution as 2d convolution"""
            conved_result = dnn.dnn_conv(img=input.dimshuffle(0, 1, 2, 'x'),
                                  kerns=self.filter_tensor.dimshuffle(0, 1, 2, 'x'),
                                  subsample=self.stride + (1,),
                                  border_mode=border_mode + (0,),
                                  conv_mode=conv_mode
                                  )[:, :, :, 0]  # drop the unused dimension

        elif len(self.convolution_axes)==2:
            conved_result = dnn.dnn_conv(img=input,
                                  kerns=self.filter_tensor,
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  conv_mode=conv_mode
                                  )
        elif len(self.convolution_axes)==3:
            conved_result = dnn.dnn_conv3d(img=input,
                                    kerns=self.filter_tensor,
                                    subsample=self.stride,
                                    border_mode=border_mode,
                                    conv_mode=conv_mode
                                    )
        else:
            raise RuntimeError("Only supports 1D, 2D and 3D convolutions")


        if any([w!=1 for w in self.width]):
            conv_output_shape = self.get_convolutional_output_shape(self.input_shape)

            target_shape = target_shape[:1] + list(self.width) + [self.num_filters] + [os//w for os,w in zip(conv_output_shape, self.width)]

            reverse_dimshuffle = [dim_shuffle_width.index(i) for i in xrange(len(dim_shuffle_width))]

            conved_result = conved_result.reshape(target_shape).dimshuffle(*reverse_dimshuffle)

            ###############
            # merge the wide kernels again with their original axis
            ###############

            target_shape = [-1, self.num_filters] + conv_output_shape
            conved_result = conved_result.reshape(target_shape)

        output_shape = list(self.input_shape)
        output_shape = [i for j, i in enumerate(output_shape) if j not in self.channel_axes and j not in self.convolution_axes]
        output_shape = [i if i is not None else -1 for i in output_shape]

        self.get_convolutional_output_shape(self.input_shape)

        result = conved_result.reshape(output_shape+
                                        list(self.filter_shape)+
                                        self.get_convolutional_output_shape(self.input_shape)
                                      )

        reverse_dimshuffle = [dim_order.index(i) for i in xrange(len(self.input_shape))]
        if reverse_dimshuffle != range(len(self.input_shape)):
            result = result.dimshuffle(*reverse_dimshuffle)

        return result




class PoolLayer(lasagne.layers.Layer):
    def __init__(self,
                 incoming,
                 pool_size=(2,2),
                 pool_axes=(2,3),
                 stride=(2,2),
                 pad=(0,0),
                 mode='max',  # alternative: 'average'
                 **kwargs):
        super(PoolLayer, self).__init__(incoming, **kwargs)
        self.pool_axes = pool_axes
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        # axis channels shrink
        for i,axis in enumerate(self.pool_axes):
            shape[axis] = lasagne.layers.pool.pool_output_length(shape[axis],
                                                                 pool_size=self.pool_size[i],
                                                                 stride=self.stride[i],
                                                                 pad=self.pad[i],
                                                                 ignore_border=True
                                                                 )
        return tuple(shape)

    def get_output_for(self, input, **kwargs):
        dimshuffle = range(len(self.input_shape))
        for axis in self.pool_axes:
            dimshuffle.remove(axis)
        for axis in self.pool_axes:
            dimshuffle.append(axis)

        new_shape = [-1,1,] + [self.input_shape[axis] for axis in self.pool_axes]

        input = input.dimshuffle(*dimshuffle).reshape(new_shape)

        pooled = dnn.dnn_pool(input, self.pool_size, self.stride, self.mode, self.pad)

        non_pool_output_shape = list(self.input_shape)
        non_pool_output_shape = [i for j, i in enumerate(non_pool_output_shape) if j not in self.pool_axes]

        pooled = pooled.reshape(replace_None(non_pool_output_shape + [self.output_shape[i] for i in self.pool_axes]))

        reverse_dimshuffle = [dimshuffle.index(i) for i in xrange(len(self.input_shape))]
        pooled = pooled.dimshuffle(*reverse_dimshuffle)
        return pooled


