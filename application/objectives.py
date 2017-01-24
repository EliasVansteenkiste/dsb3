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

    eps = 1e-15
    """
    This is the objective as defined by Kaggle: https://www.kaggle.com/c/data-science-bowl-2017/details/evaluation
    """

    def __init__(self, input_layers, target_name, *args, **kwargs):
        super(CrossEntropyObjective, self).__init__(input_layers, *args, **kwargs)
        self.target_key = target_name + ":target"
        self.target_vars[self.target_key]  = T.lvector("target_class")
        self.prediction = input_layers["predicted_probability"]

    def get_loss(self, *args, **kwargs):
        """
        Return the theano loss.
        :param args:
        :param kwargs:
        :return:
        """
        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        target_values = self.target_vars[self.target_key].astype('float32')  # convert from int
        log_loss = lasagne.objectives.binary_crossentropy(network_predictions, target_values.flatten())
        return log_loss

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        """
        Return an error where predicted and expected are numpy arrays (and not Theano)
        :param predicted:
        :param expected:
        :param args:
        :param kwargs:
        :return:
        """
        predicted = np.clip(np.array(predicted), np.float32(self.eps), np.float32(1-self.eps))
        return -np.mean(expected*np.log(predicted) + (1-expected)*np.log(1-predicted))



class SegmentationCrossEntropyObjective(TargetVarDictObjective):
    optimize = MINIMIZE
    eps = 1e-7

    def __init__(self, input_layers, target_name, *args, **kwargs):
        super(SegmentationCrossEntropyObjective, self).__init__(input_layers, *args, **kwargs)
        self.target_key = target_name + ":segmentation"
        self.target_vars[self.target_key]  = T.ftensor4("target_segmentation")
        self.prediction = input_layers["predicted_segmentation"]

    def get_loss(self, *args, **kwargs):
        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        network_predictions = np.float32(1-2*self.eps) * network_predictions + self.eps
        target_values = self.target_vars[self.target_key]
        ce = -T.log(network_predictions) * target_values - T.log(1 - network_predictions) * (1 - target_values)
        log_loss = T.mean(ce, axis=(1,2,3))
        return log_loss

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        predicted = np.float32(1-2*self.eps) * predicted + self.eps
        return -np.mean(expected*np.log(predicted) + (1-expected)*np.log(1-predicted))



class WeightedSegmentationCrossEntropyObjective(SegmentationCrossEntropyObjective):


    def __init__(self, classweights, *args, **kwargs):
        super(WeightedSegmentationCrossEntropyObjective, self).__init__(*args, **kwargs)
        self.classweights = classweights

    def get_loss(self, *args, **kwargs):

        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        target_values = self.target_vars[self.target_key]
        network_predictions = np.float32(1-2*self.eps) * network_predictions + self.eps
        ce = -T.log(network_predictions) * target_values * np.float32(self.classweights[1]) \
             -T.log(1. - network_predictions) * (1. - target_values) * np.float32(self.classweights[0])
        log_loss = T.mean(ce, axis=(1,2,3))
        return log_loss


    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        predicted = np.float32(1-2*self.eps) * predicted + self.eps
        return -np.mean(expected*np.log(predicted) * self.classweights[1] + (1-expected)*np.log(1-predicted) * self.classweights[0])