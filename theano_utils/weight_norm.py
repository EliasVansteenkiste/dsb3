import lasagne
import numpy as np
import theano
import theano.tensor as T
from theano_utils import theano_printer


class WeightNormLayer(lasagne.layers.Layer):

    layer_count = 0

    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Orthogonal("relu"), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.layer_idx= WeightNormLayer.layer_count
        WeightNormLayer.layer_count += 1
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g")

        if len(self.input_shape) == 4:
            self.axes_to_sum = (0 ,2 ,3)
            self.dimshuffle_args = ['x' ,0 ,'x' ,'x']
        elif len(self.input_shape) == 5:
            self.axes_to_sum = (0, 2, 3, 4)
            self.dimshuffle_args = ['x', 0, 'x', 'x', 'x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x' , 0]

        # scale weights in layer below
        incoming.W_param = incoming.W
        incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim == 4:
            W_axes_to_sum = (1 ,2 ,3)
            W_dimshuffle_args = [0 ,'x' ,'x' ,'x']
        elif incoming.W_param.ndim == 5:
            W_axes_to_sum = (1, 2, 3, 4)
            W_dimshuffle_args = [0, 'x', 'x', 'x', 'x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x' , 0]

        if self.g is not None:
            incoming.W = incoming.W_param * \
                              (self.g / (T.sqrt(T.sum(T.square(incoming.W_param), axis=W_axes_to_sum))) + 1e-9) \
                                  .dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / \
                              T.sqrt(T.sum(T.square(incoming.W_param), axis=W_axes_to_sum, keepdims=True))

    def get_output_for(self, input, init=-1, debug=False, **kwargs):
        self.init_updates = []
        self.apply_updates = []

        if self.layer_idx == init:
            count = theano.shared(lasagne.utils.floatX(0.))
            cum_sum = theano.shared(np.zeros(self.input_shape[1], dtype=theano.config.floatX))
            cum_sum2 = theano.shared(np.zeros(self.input_shape[1], dtype=theano.config.floatX))

            sum_ = T.mean(input, self.axes_to_sum)
            sum2 = T.mean(T.square(input), self.axes_to_sum)

            self.init_updates = [
                (cum_sum, cum_sum + sum_),
                (cum_sum2, cum_sum2 + sum2),
                (count, count + lasagne.utils.floatX(1.))
            ]
            m = cum_sum / count
            stdv = T.sqrt((cum_sum2/count - m*m)+ 1e-9) + 1e-9

            self.apply_updates = [
                (self.b, -m / stdv),
                (self.g, self.g / stdv),
                (cum_sum, T.zeros_like(cum_sum)),
                (cum_sum2, T.zeros_like(cum_sum2)),
                (count, lasagne.utils.floatX(0.))
            ]
        else:
            if hasattr(self, 'b'):
                input += self.b.dimshuffle(*self.dimshuffle_args)

        if self.layer_idx < init and debug:
        # if init==-1:
            theano_printer.print_me_this("m", T.mean(input))
            theano_printer.print_me_this("stdv", T.sqrt(T.mean(T.square(input))) + 1e-9)

        return self.nonlinearity(input)


def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)


import time
from utils.buffering import buffered_gen_threaded
from interfaces.data_loader import TRAIN
def train_weight_norm(config, output_layers, all_layers, idx, givens, xs_shared, chunk_size, required_input, required_output):
    if not hasattr(config, "init_weight_norm") or config.restart_from_save: return

    print "Initializing weights with %i samples..."%config.init_weight_norm
    t_init = time.time()

    epochs_ = config.training_data.epochs
    config.training_data.epochs = 1e-9+config.init_weight_norm*WeightNormLayer.layer_count / \
                            float(len(config.training_data.indices[TRAIN]))
    assert config.init_weight_norm*WeightNormLayer.layer_count == int(len(config.training_data.indices[TRAIN])*config.training_data.epochs)
    # print len(config.training_data.indices[TRAIN]), config.training_data.epochs
    data_generator = buffered_gen_threaded(config.training_data.generate_batch(
        chunk_size=chunk_size,
        required_input=required_input,
        required_output=required_output,
    ))

    for layer_idx in range(WeightNormLayer.layer_count):
        print "\tWN layer", layer_idx
        for network_output_layer in output_layers.values():
            lasagne.layers.helper.get_output(network_output_layer, None, init=layer_idx)
        init_updates = [u for l in all_layers for u in getattr(l, 'init_updates', [])]
        apply_updates = [u for l in all_layers for u in getattr(l, 'apply_updates', [])]
        init_wn_fun = theano.function([idx], outputs=theano_printer.get_the_stuff_to_print(),
                                      givens=givens, updates=init_updates,
                                      on_unused_input='ignore')
        apply_wn_fun = theano.function([idx], outputs=None,
                                       givens=givens, updates=apply_updates,
                                       on_unused_input='ignore')

        for i, data in enumerate(data_generator):

            for key in xs_shared:
                xs_shared[key].set_value(data["input"][key])

            for b in xrange(config.batches_per_chunk):
                init_wn_fun(b)
                # print
            # print "apply_wn_fun"
            
            # print

            num_samples = (i+1) * config.batch_size * config.batches_per_chunk
            if layer_idx != WeightNormLayer.layer_count - 1:  # let the generator die out
                if num_samples >= config.init_weight_norm: break
            # elif num_samples >= config.init_weight_norm:
            #     print num_samples
            
        apply_wn_fun(0)
        del init_wn_fun
        del apply_wn_fun

    del data_generator

    config.training_data.epochs = epochs_
    print "\ttook %.3f seconds" % (time.time() - t_init,)

