import lasagne
import theano.tensor as T
from interfaces.objectives import TargetVarDictObjective, MAXIMIZE, MINIMIZE
from theano_utils import theano_printer

import theano
import theano.tensor as T
import lasagne
import numpy as np

class CrossEntropyObjective(TargetVarDictObjective):
    optimize = MINIMIZE
    """
    This is the objective as defined by Kaggle: https://www.kaggle.com/c/data-science-bowl-2017/details/evaluation
    """

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        predicted = np.clip(np.array(predicted), np.float32(1e-15), np.float32(1-1e-15))
        return -np.mean(expected*np.log(predicted) + (1-expected)*np.log(1-predicted))

    def __init__(self, input_layers, target_name, *args, **kwargs):
        super(CrossEntropyObjective, self).__init__(input_layers, *args, **kwargs)
        self.target_key = target_name + ":class"
        self.target_vars[self.target_key]  = T.ltensor3("target_class")
        self.prediction = input_layers["predicted_probability"]

    def get_loss(self, *args, **kwargs):
        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        target_values = self.target_vars[self.target_key].astype('float32')  # convert from int
        log_loss = lasagne.objectives.binary_crossentropy(network_predictions, target_values.flatten())
        return log_loss