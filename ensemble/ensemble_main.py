import data_loading
import numpy as np
import utils_ensemble
import linear_ensemble
import utils_lung
import collections
import profile

FG_CONFIGS = ['fgodin/' + config for config in
              ['dsb_af19lme_mal2_s5_p8a1', 'dsb_af25lme_mal2_s5_p8a1', 'dsb_af4_size6_s5_p8a1',
               'dsb_af5lme_mal2_s5_p8a1', 'dsb_af5_size6_s5_p8a1', 'dsb_af24lme_mal3_s5_p8a1']]

CONFIGS = ['dsb_a04_c3ns2_mse_s5_p8a1', 'dsb_a07_c3ns3_mse_s5_p8a1', 'dsb_a08_c3ns3_mse_s5_p8a1',
           'dsb_a11_m1zm_s5_p8a1', 'dsb_af25lmeaapm_mal2_s5_p8a1',
           'dsb_a_liolme16_c3_s2_p8a1', 'dsb_a_liolme32_c3_s5_p8a1', 'dsb_a_liox10_c3_s2_p8a1',
           'dsb_a_liox11_c3_s5_p8a1', 'dsb_a_liox12_c3_s2_p8a1',
           'dsb_a_liox13_c3_s2_p8a1', 'dsb_a_liox14_c3_s2_p8a1', 'dsb_a_liox15_c3_s2_p8a1', 'dsb_a_liox6_c3_s2_p8a1',
           'dsb_a_liox7_c3_s2_p8a1', 'dsb_a_liox8_c3_s2_p8a1', 'dsb_a_liolunalme16_c3_s2_p8a1']
CONFIGS += FG_CONFIGS
OUTLIER_THRESHOLD = 0.10  # Disagreement threshold (%)
DO_MAJORITY_VOTE = False
VERBOSE = False


# TODO: to other file
class Ensemble(object):
    def __init__(self, models):
        self.models = models
        self.weights = None

    def predict(self, X):
        return self._weighted_average(X, self.weights)

    def predict_one_sample(self, X):
        assert len(X.values()[0]) == 1
        # TODO FIX BUG BECAUSE X contains only one pid
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
        return self.weights is not None


class EnsembleModel(object):
    def __init__(self, config_name, validation_preds, test_preds):
        self.config_name = config_name
        self.validation_preds = validation_preds
        self.test_preds = test_preds
        self.all_preds = validation_preds.copy()
        self.all_preds.update(test_preds)

    def predict(self, pid):
        return self.all_preds[pid]


def ensemble(configs):
    X_valid, y_valid = load_data(configs, 'validation')
    ensemble_model = linear_optimal_ensemble(X_valid, y_valid)

    X_test, y_test = load_data(configs, 'test')
    test_pids = y_test.keys()

    y_test_pred = {}

    for pid in test_pids:
        test_sample = filter_set(X_test, pid, configs)
        ensemble_pred = ensemble_model.predict_one_sample(test_sample)
        y_test_pred[pid] = majority_vote_rensemble_prediction(X_test, ensemble_pred,
                                                              pid) if DO_MAJORITY_VOTE else ensemble_pred

    test_logloss = utils_lung.evaluate_log_loss(y_test_pred, y_test)

    print 'Ensemble test logloss: ', test_logloss


def majority_vote_rensemble_prediction(X_test, ensemble_pred, pid):
    configs_to_reuse = remove_outlier_configs(X_test, ensemble_pred, pid)
    X, y = load_data(configs_to_reuse, 'validation')
    rensemble_model = linear_optimal_ensemble(X, y)
    test_sample = filter_set(X_test, pid, configs_to_reuse)
    final_pred = rensemble_model.predict_one_sample(test_sample)
    return final_pred


def remove_outlier_configs(config_predictions, ensemble_prediction, pid):
    configs_to_reuse = []
    for config in config_predictions.keys():
        config_prediction = config_predictions[config][pid]
        if ((ensemble_prediction - config_prediction) / ensemble_prediction) <= OUTLIER_THRESHOLD:
            configs_to_reuse.append(config)
    return configs_to_reuse


def filter_set(X_test, pid, configs):
    filtered_X = {}
    for config, predictions in X_test.iteritems():
        if config in configs:
            filtered_X[config] = {pid: predictions[pid]}

    return filtered_X


def load_data(configs, dataset_membership):
    if dataset_membership == 'validation':
        return data_loading.load_validation_set(configs)
    elif dataset_membership == 'test':
        return data_loading.load_test_set(configs)
    else:
        raise ValueError('Dude you drunk? No data set membership with name {} exists'.format(dataset_membership))


def linear_optimal_ensemble(predictions, labels):
    """

    :type predictions: dict
    :type labels: dict
    :param predictions: (config_name -> (pid -> prediction) )
    :param labels: ( (pid -> prediction) )
    """

    X = utils_ensemble.predictions_dict_to_3d_array(predictions)
    y = np.array(labels.values())
    weights = linear_ensemble.optimal_linear_weights(X, np.array(utils_ensemble.one_hot(y)))
    if VERBOSE: print '\nOptimal weights'
    config2weights = {}
    for model_nr in range(len(predictions.keys())):
        config = predictions.keys()[model_nr]
        if VERBOSE: print 'Weight for config {} is {:0.2%}'.format(config, weights[model_nr])
        config2weights[config] = weights[model_nr]

    ensemble_model = Ensemble(predictions.keys())
    ensemble_model.weights = config2weights
    return ensemble_model


# profile.run('ensemble(CONFIGS)')
ensemble(CONFIGS)
