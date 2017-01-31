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





class VolumeSegmentationObjective(TargetVarDictObjective):

    def __init__(self, input_layer, target_name, *args, **kwargs):
        super(VolumeSegmentationObjective, self).__init__(input_layer, *args, **kwargs)
        self.target_key = target_name
        self.target_vars[self.target_key]  = T.ftensor4("target_segmentation")
        self.prediction = input_layer

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        raise NotImplementedError()


class SegmentationCrossEntropyObjective(VolumeSegmentationObjective):
    optimize = MINIMIZE
    eps = 1e-7

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
    """
    Weighted cross entropy. One class is weighted according with other class
    """
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




class JaccardIndexObjective(VolumeSegmentationObjective):
    """
    Jaccard Index: https://en.wikipedia.org/wiki/Jaccard_index
    """
    optimize = MAXIMIZE

    def __init__(self, smooth=1., *args, **kwargs):
        super(JaccardIndexObjective, self).__init__(*args, **kwargs)
        self.smooth = np.float32(smooth)

    def get_loss(self, *args, **kwargs):
        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        target_values = self.target_vars[self.target_key]

        y_true_f = target_values
        y_pred_f = network_predictions
        intersection = T.sum(y_true_f * y_pred_f, axis=(1,2,3))
        return (intersection + self.smooth) / (T.sum(y_true_f, axis=(1,2,3)) + T.sum(y_pred_f, axis=(1,2,3)) - intersection + self.smooth)

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        y_true_f = expected.flatten()
        y_pred_f = predicted.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (intersection + self.smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + self.smooth)



class SoerensonDiceCoefficientObjective(VolumeSegmentationObjective):
    """
    Soerensen-Dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """
    optimize = MAXIMIZE

    def __init__(self, smooth=1., *args, **kwargs):
        super(SoerensonDiceCoefficientObjective, self).__init__(*args, **kwargs)
        self.smooth = np.float32(smooth)

    def get_loss(self, *args, **kwargs):
        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        target_values = self.target_vars[self.target_key]

        y_true_f = target_values
        y_pred_f = network_predictions

        intersection = T.sum(y_true_f * y_pred_f, axis=(1,2,3))
        return ( 2 * intersection + self.smooth) / (T.sum(y_true_f, axis=(1,2,3)) + T.sum(y_pred_f, axis=(1,2,3)) + self.smooth)

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        y_true_f = expected.flatten()
        y_pred_f = predicted.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2 * intersection + self.smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + self.smooth)


class PrecisionObjective(VolumeSegmentationObjective):
    """
    Precision: https://en.wikipedia.org/wiki/Precision_and_recall
    """

    optimize = MAXIMIZE

    def __init__(self, smooth=1., *args, **kwargs):
        super(PrecisionObjective, self).__init__(*args, **kwargs)
        self.smooth = smooth

    def get_loss(self, *args, **kwargs):
        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        target_values = self.target_vars[self.target_key]

        y_true_f = target_values
        y_pred_f = network_predictions

        true_positive = T.sum(y_true_f * y_pred_f, axis=(1,2,3))
        false_positive = T.sum((1.-y_true_f) * y_pred_f, axis=(1,2,3))

        return (true_positive + self.smooth) / (true_positive + false_positive + self.smooth)

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        y_true_f = expected.flatten()
        y_pred_f = predicted.flatten()

        true_positive = np.sum(y_true_f * y_pred_f)
        false_positive = np.sum((1.-y_true_f) * y_pred_f)

        return (true_positive + self.smooth) / (true_positive + false_positive + self.smooth)


class RecallObjective(VolumeSegmentationObjective):
    """
    Recall: https://en.wikipedia.org/wiki/Precision_and_recall
    """
    optimize = MAXIMIZE

    def __init__(self, smooth=1., *args, **kwargs):
        super(RecallObjective, self).__init__(*args, **kwargs)
        self.smooth = smooth

    def get_loss(self, *args, **kwargs):
        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        target_values = self.target_vars[self.target_key]

        y_true_f = target_values
        y_pred_f = network_predictions

        true_positive = T.sum(y_true_f * y_pred_f, axis=(1,2,3))
        false_negative = T.sum(y_true_f * (1.-y_pred_f), axis=(1,2,3))

        return (true_positive + self.smooth) / (true_positive + false_negative + self.smooth)

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        y_true_f = expected.flatten()
        y_pred_f = predicted.flatten()

        true_positive = np.sum(y_true_f * y_pred_f)
        false_negative = np.sum(y_true_f * (1.-y_pred_f))

        return (true_positive + self.smooth) / (true_positive + false_negative + self.smooth)


class ClippedFObjective(VolumeSegmentationObjective):
    """
    Recall: https://en.wikipedia.org/wiki/Precision_and_recall
    """
    optimize = MAXIMIZE

    def __init__(self, smooth=1., recall_weight=1.0, precision_weight=1.0, *args, **kwargs):
        super(ClippedFObjective, self).__init__(*args, **kwargs)
        self.smooth = smooth
        self.recall_weight = recall_weight
        self.precision_weight = precision_weight

    def get_loss(self, *args, **kwargs):
        network_predictions = lasagne.layers.helper.get_output(self.prediction, *args, **kwargs)
        target_values = self.target_vars[self.target_key]

        y_true_f = target_values
        y_pred_f = network_predictions

        true_positive = T.sum(y_true_f * y_pred_f, axis=(1,2,3))
        false_negative = T.sum(y_true_f * (1.-y_pred_f), axis=(1,2,3))
        false_positive = T.sum((1.-y_true_f) * y_pred_f, axis=(1,2,3))

        recall = (true_positive + self.smooth) / (true_positive + false_negative + self.smooth)
        precision = (true_positive + self.smooth) / (true_positive + false_positive + self.smooth)

        return T.minimum(recall*self.recall_weight, 1.0) * 0.5 + T.minimum(precision*self.precision_weight, 1.0) * 0.5

    def get_loss_from_lists(self, predicted, expected, *args, **kwargs):
        y_true_f = expected.flatten()
        y_pred_f = predicted.flatten()

        true_positive = np.sum(y_true_f * y_pred_f)
        false_negative = np.sum(y_true_f * (1.-y_pred_f))
        false_positive = np.sum((1.-y_true_f) * y_pred_f)

        recall = (true_positive + self.smooth) / (true_positive + false_negative + self.smooth)
        precision = (true_positive + self.smooth) / (true_positive + false_positive + self.smooth)

        return np.minimum(recall*self.recall_weight, 1.0) * 0.5 + np.minimum(precision*self.precision_weight, 1.0) * 0.5
