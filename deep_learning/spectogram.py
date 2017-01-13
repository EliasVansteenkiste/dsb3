import numpy as np
import theano.tensor as T
import lasagne

import theano
from theano.sandbox.cuda import dnn
from scipy import signal

if not theano.sandbox.cuda.cuda_enabled:
    raise ImportError(
            "requires GPU support -- see http://lasagne.readthedocs.org/en/"
            "latest/user/installation.html#gpu-support")  # pragma: no cover
elif not dnn.dnn_available():
    raise ImportError(
            "cuDNN not available: %s\nSee http://lasagne.readthedocs.org/en/"
            "latest/user/installation.html#cudnn" %
            dnn.dnn_available.msg)  # pragma: no cover

class SpectogramLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, window_size=None, stride=1, f=lasagne.init.Constant(1.), nonlinearity=T.log, *args, **kwargs):
        super(SpectogramLayer, self).__init__(incoming, *args, **kwargs)
        self.num_units = num_units

        self.f = self.add_param(f, (num_units,), name="f")

        self.window_size = window_size
        self.window = signal.tukey(self.window_size, alpha=0.25).astype('float32')
        self.window = self.window * self.window_size / np.sum(self.window)

        self.stride = stride
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    @property
    def filter_tensor(self):
        #(self.num_units*2, window)
        t = np.arange(start=0, stop=self.window_size, dtype='float32')
        two_pi = np.float32(2*np.pi)
        ph = t[None,None,:]*two_pi/self.f[:,None,None]
        cosines = 2*T.cos(ph)*(self.window/np.float32(self.window_size))
        sines   = 2*T.sin(ph)*(self.window/np.float32(self.window_size))

        W = T.concatenate([cosines, sines], axis=1)
        self.W = W.reshape(shape=(self.num_units*2,self.window_size))
        return self.W


    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        return (shape[0], shape[1], self.num_units, (input_shape[2]-self.window_size+1)/self.stride)


    def get_output_for(self, input, **kwargs):
        input = input.reshape(shape=(-1, self.input_shape[2]))

        conved = dnn.dnn_conv(img=input.dimshuffle(0, 'x', 1, 'x'),
                              kerns=self.filter_tensor.dimshuffle(0, 'x', 1, 'x'),
                              subsample=(self.stride, 1),
                              border_mode=(0, 0),
                              conv_mode='cross'
                              )

        conved = conved.reshape(shape=(-1, self.input_shape[1], self.num_units*2, conved.shape[2]))
        result = conved[:,:,0::2,:]**2 + conved[:,:,1::2,:]**2
        return self.nonlinearity(result)


if __name__=="__main__":
    inp = lasagne.layers.InputLayer(shape=(1,1,128*128))
    spect = SpectogramLayer(inp, num_units=4, f=np.array([50,100,200,400], dtype='float32'), window_size=128*128/4, stride=128*128/4, nonlinearity=None)
    outp = lasagne.layers.helper.get_output(spect)
    func = theano.function([inp.input_var],[outp])
    t = np.arange(start=0, stop=128*128/4)
    tp = 2*np.pi
    signal = np.concatenate((np.cos(t*tp/50),
                             np.cos(t*tp/100),
                             np.cos(t*tp/200),
                             np.cos(t*tp/400)))
    result, = func(signal[None,None,:].astype('float32'))
    print result





