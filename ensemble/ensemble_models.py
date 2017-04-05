import theano
import theano.tensor as T

import utils_lung
from ensemble import utils_ensemble
import numpy as np
import scipy.optimize
import collections

VERBOSE = False


class WeightedEnsemble(object):
    def __init__(self, models, optimization_method):
        self.models = models
        self.weights = {}
        self.optimization_method = optimization_method
        self.training_error = None

    def train(self, predictions, labels):
        X = utils_ensemble.predictions_dict_to_3d_array(predictions)
        y = np.array(labels.values())

        weights = self.optimization_method(X, np.array(utils_ensemble.one_hot(y)))
        for model_nr in range(len(self.models)):
            config = self.models[model_nr]
            self.weights[config] = weights[model_nr]

        y_pred = self._weighted_average(predictions, self.weights)
        self.training_error = utils_lung.evaluate_log_loss(y_pred, labels)

    def predict(self, X):
        return self._weighted_average(X, self.weights)

    def predict_one_sample(self, X):
        assert len(X.values()[0]) == 1
        return self._weighted_average(X, self.weights).values()[0]

    def _weighted_average(self, predictions, weights):
        """
        Arithmetic average

        :param predictions: (config_name -> (pid -> prediction) )
        :param weights: (config_name -> (weight) )
        :return  predictions as dict: (pid -> prediction)
        """
        if not self.__is_optimized():
            raise ValueError('Cannot use ensemble if not optimized')
        weighted_predictions = {}
        for config_name, config_predictions in predictions.iteritems():
            for pid, patient_prediction in config_predictions.iteritems():
                weighted_prediction = patient_prediction * weights[config_name]
                if pid in weighted_predictions:
                    weighted_predictions[pid] += weighted_prediction
                else:
                    weighted_predictions[pid] = weighted_prediction

        return collections.OrderedDict(sorted(weighted_predictions.items()))

    def __is_optimized(self):
        return self.weights is not None and len(self.weights) != 0


class WeightedMajorityVoteEnsemble(WeightedEnsemble):
    def __init__(self, models, optimization_method):
        super(WeightedMajorityVoteEnsemble, self).__init__(models, optimization_method)


class EqualWeightsOfTopRankingModelsEnsemble(WeightedEnsemble):
    def __init__(self, models, optimization_method):
        super(EqualWeightsOfTopRankingModelsEnsemble, self).__init__(models, optimization_method)


def linear_optimal_ensemble(predictions, labels):
    """

    :type predictions: dict
    :type labels: dict
    :param predictions: (config_name -> (pid -> prediction) )
    :param labels: ( (pid -> prediction) )
    """
    X = utils_ensemble.predictions_dict_to_3d_array(predictions)
    y = np.array(labels.values())
    weights = optimal_linear_weights(X, np.array(utils_ensemble.one_hot(y)))
    if VERBOSE: print '\nOptimal weights'
    config2weights = {}
    for model_nr in range(len(predictions.keys())):
        config = predictions.keys()[model_nr]
        if VERBOSE: print 'Weight for config {} is {:0.2%}'.format(config, weights[model_nr])
        config2weights[config] = weights[model_nr]

    ensemble_model = WeightedEnsemble(predictions.keys())
    ensemble_model.weights = config2weights
    return ensemble_model


def optimal_linear_weights(predictions_stack, targets):
    """
    :param predictions_stack:  predictions as numpy array with shape [num_configs x num_patients x 2]
    :param targets: target labels as one hot encoded 2D array with shape [num_patients x 2]
    :return:
    """
    X = theano.shared(predictions_stack.astype(theano.config.floatX))  # [num_configs x num_patients x 2]
    t = theano.shared(targets)
    W = T.vector('W')
    s = T.nnet.softmax(W).reshape((W.shape[0], 1, 1))
    weighted_avg_predictions = T.sum(X * s, axis=0)  # T.tensordot(X, s, [[0], [0]])
    error = utils_ensemble.log_loss(weighted_avg_predictions, t)
    grad = T.grad(error, W)
    f = theano.function([W], error)
    g = theano.function([W], grad)
    n_models = predictions_stack.shape[0]
    w_init = np.zeros(n_models, dtype=theano.config.floatX)
    out, loss, _ = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)
    weights = np.exp(out)
    weights /= weights.sum()
    return weights


def equal_weights(predictions_stack, targets):
    amount_of_configs = predictions_stack.shape[0]
    equal_weight = 1.0 / amount_of_configs

    weights = [equal_weight for _ in range(amount_of_configs)]
    return weights
