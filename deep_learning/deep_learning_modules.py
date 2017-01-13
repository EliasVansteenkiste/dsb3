


def residual(incoming, num_filters=None,
              num_conv=3,
              filter_size=(3,3), pool_size=(2,2), pad=(1,1), channel=1, axis=(2,3),
              W=nn.init.Orthogonal("relu"), b=nn.init.Constant(0.0),
              nonlinearity=nn.nonlinearities.rectify):

    l_h = incoming

    for _ in xrange(num_conv):
        l_h = ConvolutionOver2DAxisLayer(l_h, num_filters=num_filters,
                                          axis=axis, channel=channel,
                                            filter_size=filter_size,
                                          pad=pad,
                                            W=W, b=b,
                                            nonlinearity=nonlinearity)

    l_maxpool = MaxPoolOver2DAxisLayer(l_h, pool_size=pool_size,
                                          stride=pool_size,
                                          axis=axis)
    # reduce the incoming layers size to more or less the remaining size after the
    # previous steps, but with the correct number of channels
    l_maxpool_incoming = MaxPoolOver2DAxisLayer(incoming, pool_size=pool_size,
                                                          stride=pool_size,
                                                          axis=axis)

    l_proc_incoming = PadWithZerosLayer(l_maxpool_incoming,
                                        final_size=num_filters
                                        )

    return MultiplicativeGatingLayer(gate=None, input1=l_maxpool, input2=l_proc_incoming)