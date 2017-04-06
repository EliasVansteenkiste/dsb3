import utils
import data_loading
import numpy as np
import os
import evaluate_submission
import utils_ensemble
import ensemble_analysis as anal
import ensemble_models as em
import utils_lung
import collections
import profile
from sklearn.model_selection import StratifiedKFold

FG_CONFIGS = ['fgodin/' + config for config in
              ['dsb_af19lme_mal2_s5_p8a1', 'dsb_af25lme_mal2_s5_p8a1', 'dsb_af4_size6_s5_p8a1',
               'dsb_af5lme_mal2_s5_p8a1', 'dsb_af5_size6_s5_p8a1', 'dsb_af24lme_mal3_s5_p8a1']]

EV_CONFIGS = ['eavsteen/' + config for config in ['dsb_a_eliasq1_mal2_s5_p8a1']]

CONFIGS = ['dsb_a04_c3ns2_mse_s5_p8a1', 'dsb_a07_c3ns3_mse_s5_p8a1', 'dsb_a08_c3ns3_mse_s5_p8a1',
           'dsb_a11_m1zm_s5_p8a1', 'dsb_af25lmeaapm_mal2_s5_p8a1',
           'dsb_a_liolme16_c3_s2_p8a1', 'dsb_a_liolme32_c3_s5_p8a1', 'dsb_a_liox10_c3_s2_p8a1',
           'dsb_a_liox11_c3_s5_p8a1', 'dsb_a_liox12_c3_s2_p8a1', 'dsb_af25lmelr10-2_mal2_s5_p8a1',
           'dsb_af25lmelr10-1_mal2_s5_p8a1', 'dsb_a_eliasz1_c3_s5_p8a1',
           'dsb_a_liox13_c3_s2_p8a1', 'dsb_a_liox14_c3_s2_p8a1', 'dsb_a_liox15_c3_s2_p8a1', 'dsb_a_liox6_c3_s2_p8a1',
           'dsb_a_liox7_c3_s2_p8a1', 'dsb_a_liox8_c3_s2_p8a1', 'dsb_a_liolunalme16_c3_s2_p8a1',
           'dsb_a_lionoclip_c3_s5_p8a1', 'dsb_a_liomse_c3_s5_p8a1', 'dsb_af25lmeo0_s5_p8a1', 'dsb_af25lmelr10-3_mal2_s5_p8a1']

FG_CONFIGS = ['fgodin/' + config for config in ['dsb_af25lme_mal2_s5_p8a1']]
GOOD_CONFIGS = ['dsb_af25lmeaapm_mal2_s5_p8a1', 'dsb_a_liolme32_c3_s5_p8a1', 'dsb_af25lmelr10-1_mal2_s5_p8a1',
                'dsb_a_liox10_c3_s2_p8a1']

CONFIGS = FG_CONFIGS + CONFIGS + EV_CONFIGS
OUTLIER_THRESHOLD = 0.10  # Disagreement threshold (%)
DO_MAJORITY_VOTE = False
DO_CV = False
VERBOSE = True
expid = utils.generate_expid('ensemble')


def aggressive_ensembling(configs):
    """
    Take models trained on all the data. Do a cross validation to get a ranking between the models. Choose the top N models.
    Merge these top N model into an equally weighted model.
    """
    X_valid, y_valid = load_data(configs, 'validation')
    anal.analyse_predictions(X_valid, y_valid)

    cv = do_cross_validation(X_valid, y_valid, configs, em.optimal_linear_weights)
    if DO_CV:
        anal.analyse_cv_result(cv, 'linear optimal weight')
        anal.analyse_cv_result(do_cross_validation(X_valid, y_valid, configs, em.equal_weights), 'equal weight')

    configs_to_use = prune_configs(configs, cv)
    print 'final ensemble will use configs: ', configs_to_use
    ensemble_model = em.WeightedEnsemble(configs_to_use, optimization_method=em.equal_weights)
    ensemble_model.train(X_valid, y_valid)
    print 'Ensemble training error: ', ensemble_model.training_error
    ensemble_model.print_weights()

    X_test, y_test = load_data(configs_to_use, 'test')
    test_pids = y_test.keys()

    y_test_pred = {}

    for pid in test_pids:
        test_sample = filter_set(X_test, pid, configs_to_use)
        ensemble_pred = ensemble_model.predict_one_sample(test_sample)
        y_test_pred[pid] = majority_vote_rensemble_prediction(X_test, ensemble_pred,
                                                              pid) if DO_MAJORITY_VOTE else ensemble_pred

    evaluate_test_set_performance(y_test, y_test_pred)
    utils_ensemble.persist_test_set_predictions(expid, y_test_pred)


def conservative_ensembling(configs):
    """
    Take models trained on training data. Optimise the hell out of it using the validation data.
    This is to protect against overfitted or very bad models.
    """
    X_valid, y_valid = load_data(configs, 'validation')
    anal.analyse_predictions(X_valid, y_valid)

    if DO_CV:
        anal.analyse_cv_result(do_cross_validation(X_valid, y_valid, configs, em.optimal_linear_weights),
                               'linear optimal weight')
        anal.analyse_cv_result(do_cross_validation(X_valid, y_valid, configs, em.equal_weights), 'equal weight')

    ensemble_model = em.WeightedEnsemble(configs, optimization_method=em.optimal_linear_weights)
    ensemble_model.train(X_valid, y_valid)
    print 'Ensemble training error: ', ensemble_model.training_error
    ensemble_model.print_weights()

    X_test, y_test = load_data(configs, 'test')
    test_pids = y_test.keys()

    y_test_pred = {}

    for pid in test_pids:
        test_sample = filter_set(X_test, pid, configs)
        ensemble_pred = ensemble_model.predict_one_sample(test_sample)
        y_test_pred[pid] = majority_vote_rensemble_prediction(X_test, ensemble_pred,
                                                              pid) if DO_MAJORITY_VOTE else ensemble_pred

    evaluate_test_set_performance(y_test, y_test_pred)
    utils_ensemble.persist_test_set_predictions(expid, y_test_pred)


def prune_configs(configs_used, cv_result):
    # prune if a config was never used during all the folds of CV.
    configs_that_are_never_used = list(configs_used)
    for cv in cv_result:
        weights = cv['weights']
        config_names = np.array(cv['configs'])

        unused_configs = config_names[(np.isclose(weights, np.zeros_like(weights)))]
        for c in configs_that_are_never_used:
            if c not in unused_configs:
                configs_that_are_never_used.remove(c)

    return [config for config in configs_used if config not in configs_that_are_never_used]


def evaluate_test_set_performance(y_test, y_test_pred):
    test_set_predictions = {config: data_loading.get_predictions_of_config(config, 'test') for config in CONFIGS}
    individual_performance = {config: calc_test_performance(config, pred_test) for config, pred_test in
                              test_set_predictions.iteritems()}
    for config, performance in individual_performance.iteritems():
        print 'Logloss of config {} is {} on test set'.format(config, performance)
    test_logloss = utils_lung.evaluate_log_loss(y_test_pred, y_test)
    print 'Ensemble test logloss: ', test_logloss


def do_cross_validation(X, y, config_names, ensemble_method=em.optimal_linear_weights):
    X = utils_ensemble.predictions_dict_to_3d_array(X)
    y = np.array(y.values())

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, random_state=0)
    cv_result = []
    for train_index, test_index in skf.split(np.zeros(y.shape[0]), y):
        if np.any([test_sample in train_index for test_sample in test_index]):
            raise ValueError('\n---------------\nData leak!\n---------------\n')

        X_train, X_test = X[:, train_index, :], X[:, test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        weights = ensemble_method(X_train, np.array(utils_ensemble.one_hot(y_train)))

        y_train_pred = np.zeros(len(train_index))
        y_test_pred = np.zeros(len(test_index))
        for i, weight in enumerate(weights):
            y_train_pred += X_train[i, :, 1] * weights[i]  # this can probably be replaced with a tensor dot product
            y_test_pred += X_test[i, :, 1] * weights[i]  # this can probably be replaced with a tensor dot product

        training_loss = utils_lung.log_loss(y_train, y_train_pred)
        valid_loss = utils_lung.log_loss(y_test, y_test_pred)
        cv_result.append({
            'weights': weights,
            'training_loss': training_loss,
            'validation_loss': valid_loss,
            'training_idx': train_index,
            'test_idx': test_index,
            'configs': config_names,
            'ensemble_method': ensemble_method
        })

    return cv_result


def majority_vote_rensemble_prediction(X_test, ensemble_pred, pid):
    configs_to_reuse = remove_outlier_configs(X_test, ensemble_pred, pid)
    X, y = load_data(configs_to_reuse, 'validation')
    # rensemble_model = em.linear_optimal_ensemble(X, y)
    rensemble_model = em.WeightedEnsemble(configs_to_reuse, em.equal_weights)
    rensemble_model.train(X, y)
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


def calc_test_performance(config_name, predictions):
    config_name = config_name.replace('/', '')
    tmp_submission_file = '/tmp/submission_test_predictions_{}.csv'.format(config_name)
    utils_lung.write_submission(predictions, tmp_submission_file)
    loss = evaluate_submission.leaderboard_performance(tmp_submission_file)
    os.remove(tmp_submission_file)
    return loss


print 'Starting ensemble procedure with {} configs'.format(len(CONFIGS))
# aggressive_ensembling(CONFIGS)
conservative_ensembling(CONFIGS)
