import theano.tensor

def last_dim_softmax(x):
    x_shape = x.shape
    x = x.reshape((-1,x.shape[-1]))
    out = theano.tensor.nnet.softmax(x)
    out = out.reshape(x_shape)
    return out
