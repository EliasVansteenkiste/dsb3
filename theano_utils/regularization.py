import theano.tensor as T

def L1(x):
    """Computes the L1 norm of a tensor
    Parameters
    ----------
    x : Theano tensor
    Returns
    -------
    Theano scalar
        l1 norm (sum of absolute values of elements)
    """
    return T.sum(abs(x))#,axis=range(1,x.ndim))


def L2(x):
    """Computes the squared L2 norm of a tensor
    Parameters
    ----------
    x : Theano tensor
    Returns
    -------
    Theano scalar
        squared l2 norm (sum of squared values of elements)
    """
    return T.sum(x**2)#,axis=range(1,x.ndim))