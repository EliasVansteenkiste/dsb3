import data_loading
import numpy as np
import utils_ensemble
import ensemble_models as em
import utils_lung
import collections
import profile

FG_CONFIGS = ['fgodin/' + config for config in
              ['dsb_af19lme_mal2_s5_p8a1', 'dsb_af25lme_mal2_s5_p8a1', 'dsb_af4_size6_s5_p8a1',
               'dsb_af5lme_mal2_s5_p8a1', 'dsb_af5_size6_s5_p8a1', 'dsb_af24lme_mal3_s5_p8a1']]

CONFIGS = ['dsb_a04_c3ns2_mse_s5_p8a1', 'dsb_a07_c3ns3_mse_s5_p8a1', 'dsb_a08_c3ns3_mse_s5_p8a1',
           'dsb_a11_m1zm_s5_p8a1', 'dsb_af25lmeaapm_mal2_s5_p8a1',
           'dsb_a_liolme16_c3_s2_p8a1', 'dsb_a_liolme32_c3_s5_p8a1', 'dsb_a_liox10_c3_s2_p8a1',
           'dsb_a_liox11_c3_s5_p8a1', 'dsb_a_liox12_c3_s2_p8a1', 'dsb_af25lmelr10-2_mal2_s5_p8a1',
           'dsb_af25lmelr10-1_mal2_s5_p8a1', 'dsb_a_eliasz1_c3_s5_p8a1',
           'dsb_a_liox13_c3_s2_p8a1', 'dsb_a_liox14_c3_s2_p8a1', 'dsb_a_liox15_c3_s2_p8a1', 'dsb_a_liox6_c3_s2_p8a1',
           'dsb_a_liox7_c3_s2_p8a1', 'dsb_a_liox8_c3_s2_p8a1', 'dsb_a_liolunalme16_c3_s2_p8a1']
CONFIGS += FG_CONFIGS
OUTLIER_THRESHOLD = 0.20  # Disagreement threshold (%)
DO_MAJORITY_VOTE = False
VERBOSE = True


def ensemble(configs):
    X_valid, y_valid = load_data(configs, 'validation')
    ensemble_model = em.linear_optimal_ensemble(X_valid, y_valid)

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
    rensemble_model = em.linear_optimal_ensemble(X, y)
    test_sample = filter_set(X_test, pid, configs_to_reuse)
    final_pred = rensemble_model.predict_one_sample(test_sample)
    return final_pred


def remove_outlier_configs(config_predictions, ensemble_prediction, pid):
    relative_diff = False
    configs_to_reuse = []
    for config in config_predictions.keys():
        config_prediction = config_predictions[config][pid]
        diff = ((ensemble_prediction - config_prediction) / ensemble_prediction) \
            if relative_diff else abs(ensemble_prediction - config_prediction)
        if diff <= OUTLIER_THRESHOLD:
            configs_to_reuse.append(config)
        elif VERBOSE:
            print 'Removing config ', config, ' from ensemble'
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


# profile.run('ensemble(CONFIGS)')

ensemble(CONFIGS)
