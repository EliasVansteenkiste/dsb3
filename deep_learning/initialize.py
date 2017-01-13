from lasagne.init import Initializer
from lasagne.utils import floatX
import numpy as np

class Eye(Initializer):
    """Intialize weights as Orthogonal matrix.

    Orthogonal matrix initialization [1]_. For n-dimensional shapes where
    n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
    corresponds to the fan-in, so this makes the initialization usable for
    both dense and convolutional layers.

    Parameters
    ----------
    gain : float or 'relu'
        Scaling factor for the weights. Set this to ``1.0`` for linear and
        sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
        to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
        leakiness ``alpha``. Other transfer functions may need different
        factors.

    References
    ----------
    .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
           "Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks." arXiv preprint arXiv:1312.6120 (2013).
    """
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.gain = gain

    def sample(self, shape):
        if len(shape) != 2:
            raise RuntimeError("Only shapes of length 2 are "
                               "supported.")
        q = np.eye(*shape)
        return floatX(self.gain * q)