"""
This script ensembles predictions of all the models. No bagging atm. just plain simple averaging to get started.
Final predictions are weighted average of all predictions. The weights are optimized on validation data.
Mind that the terms 'models' and 'configs' are used interchangeably
"""
import numpy as np
import scipy
import theano
import theano.tensor as T
import collections
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import utils_ensemble
import utils_plots
import matplotlib.pyplot as plt
import scipy.stats

import evaluate_submission
import pathfinder
import utils
import utils_lung
import os.path as path
import os
import sandbox.ensemble_analysis as analyse

FG_CONFIGS = ['fgodin/' + config for config in
              ['dsb_af19lme_mal2_s5_p8a1', 'dsb_af25lme_mal2_s5_p8a1', 'dsb_af4_size6_s5_p8a1',
               'dsb_af5lme_mal2_s5_p8a1', 'dsb_af5_size6_s5_p8a1', 'dsb_af24lme_mal3_s5_p8a1']]

CONFIGS = ['dsb_a04_c3ns2_mse_s5_p8a1', 'dsb_a07_c3ns3_mse_s5_p8a1', 'dsb_a08_c3ns3_mse_s5_p8a1',
           'dsb_a11_m1zm_s5_p8a1', 'dsb_a_liox8_c3_s2_p8a1', 'dsb_a_liolme16_c3_s2_p8a1', 'dsb_a_liox6_c3_s2_p8a1',
           'dsb_a_liox7_c3_s2_p8a1']

# FG_CONFIGS = ['fgodin/' + config for config in
#               ['dsb_af19lme_mal2_s5_p8a1', 'dsb_af25lme_mal2_s5_p8a1', 'dsb_af5lme_mal2_s5_p8a1']]
#
# CONFIGS = ['dsb_a_liox8_c3_s2_p8a1', 'dsb_a_liolme16_c3_s2_p8a1', 'dsb_a_liox7_c3_s2_p8a1', ]

CONFIGS += FG_CONFIGS

expid = utils.generate_expid('ensemble')


def linear_stacking():
    valid_set_predictions, valid_set_labels = load_validation_set()
    analyse.analyse_predictions(valid_set_predictions, valid_set_labels)

    weights = optimize_weights(valid_set_predictions, valid_set_labels)  # (config_name -> (weight) )

    y_valid_pred = weighted_average(valid_set_predictions, weights)

    test_set_predictions = {config: get_predictions_of_config(config, 'test') for config in CONFIGS}
    y_test_pred = weighted_average(test_set_predictions, weights)
    utils_ensemble.persist_predictions(y_test_pred, y_valid_pred, expid)
    compare_test_performance_ind_vs_ensemble(test_set_predictions)


def tree_stacking():
    valid_set_predictions, valid_set_labels = load_validation_set()
    analyse.analyse_predictions(valid_set_predictions, valid_set_labels)

    X = predictions_dict_to_3d_array(valid_set_predictions, valid_set_labels)
    y = np.array(valid_set_labels.values())
    config_names = valid_set_predictions.keys()

    cv_results = {}

    cv_results['tree_stacking'] = gradient_boosting(X, y)

    model = cv_results['tree_stacking']['model']

    test_set_predictions = {config: get_predictions_of_config(config, 'test') for config in CONFIGS}
    X_test = predictions_dict_to_3d_array(test_set_predictions, None)

    y_valid_pred = model.predict_proba(X[:, :, 1].T)[:, 1]
    y_test_pred = {test_set_predictions.values()[0].keys()[i]: pred for i, pred in
                   enumerate(model.predict_proba(X_test[:, :, 1].T)[:, 1])}
    utils_ensemble.persist_predictions(y_test_pred, y_valid_pred, expid)
    compare_test_performance_ind_vs_ensemble(test_set_predictions)


def load_validation_set():
    valid_set_predictions = collections.OrderedDict()  # (config_name -> (pid -> prediction) )
    for config in CONFIGS:
        valid_set_predictions[config] = get_predictions_of_config(config, 'valid')
    valid_set_labels = load_validation_labels()  # (pid -> prediction)
    sanity_check(valid_set_predictions, valid_set_labels)
    return valid_set_predictions, valid_set_labels


def get_predictions_of_config(config_name, which_set):
    predictions_dir = os.path.join(pathfinder.METADATA_PATH, 'model-predictions')
    exp_id = utils_ensemble.find_model_preds_expid(predictions_dir, config_name)

    output_pkl_file = predictions_dir + '/%s-%s-%s.pkl' % (config_name, exp_id, which_set)
    preds = utils.load_pkl(output_pkl_file)  # pid2prediction
    preds = collections.OrderedDict(sorted(preds.items()))
    # print 'Passing predicions from {}'.format(output_pkl_file)
    return preds


def load_validation_labels():
    train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
    valid_pids = train_valid_ids['validation']
    id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)

    labels = {pid: id2label[pid] for pid in sorted(valid_pids)}
    return collections.OrderedDict(sorted(labels.items()))


def sanity_check(valid_set_predictions, valid_set_labels):
    for config in CONFIGS:
        # Check whether all these configs contain all the predictions
        if valid_set_predictions[config].viewkeys() != valid_set_labels.viewkeys():
            raise ValueError(
                'the validation set predictions does not contain the same pids as the validation set labels')

    pass


def optimize_weights(predictions, labels):
    """

    :type predictions: dict
    :type labels: dict
    :param predictions: (config_name -> (pid -> prediction) )
    :param labels: ( (pid -> prediction) )
    :return  optimized weights as dict: (config_name -> (weight) )
    """
    print 'Optimizing weights...'
    X = predictions_dict_to_3d_array(predictions, labels)
    y = np.array(labels.values())
    config_names = predictions.keys()

    cv_results = {}

    cv_results['optimal_linear_weights'] = do_cross_validation(X, y, config_names, optimal_linear_weights)
    cv_results['equal_weights'] = do_cross_validation(X, y, config_names, simple_average)

    analyse.analyse_cv_result(cv_results['optimal_linear_weights'], 'optimal_linear_weights')
    analyse.analyse_cv_result(cv_results['equal_weights'], 'equal_weights')

    best_cv_method = None
    lowest_loss = np.inf
    for cv_result in cv_results.values():
        losses = np.array([cv['validation_loss'] for cv in cv_result])
        loss = np.mean(losses)
        if loss <= lowest_loss:
            lowest_loss = loss
            best_cv_method = cv_result[0]['ensemble_method']

    print 'Model with best CV result is ', best_cv_method.func_name
    # weights = optimal_linear_weights(X, np.array(utils_ensemble.one_hot(y)))
    # weights = simple_average(X, np.array(utils_ensemble.one_hot(y)))

    weights = best_cv_method(X, np.array(utils_ensemble.one_hot(y)))

    print 'Optimal weights'
    config2weights = {}
    for model_nr in range(len(predictions.keys())):
        config = predictions.keys()[model_nr]
        print 'Weight for config {} is {:0.2%}'.format(config, weights[model_nr])
        config2weights[config] = weights[model_nr]

    return config2weights


def do_cross_validation(X, y, config_names, ensemble_method):
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


def predictions_dict_to_3d_array(predictions, labels):
    """
    :param predictions: (config_name -> (pid -> prediction) )
    :param labels: ( (pid -> prediction) )
    :return: predictions as numpy array with shape [num_configs x num_patients x 2]
    """
    n_models = len(predictions.keys())
    pids = predictions.values()[0].keys()
    n_patients = len(pids)
    predictions_stack = np.zeros((n_models, n_patients, 2))  # num_configs x num_patients x 2 categories
    for model_nr in range(n_models):
        config = predictions.keys()[model_nr]
        for patient_nr, patient_id in enumerate(pids):
            predictions_stack[model_nr, patient_nr, 0] = 1.0 - predictions[config][patient_id]
            predictions_stack[model_nr, patient_nr, 1] = predictions[config][patient_id]
    return predictions_stack


def simple_average(predictions_stack, targets):
    amount_of_configs = predictions_stack.shape[0]
    equal_weight = 1.0 / amount_of_configs

    weights = [equal_weight for _ in range(amount_of_configs)]
    return weights


def gradient_boosting(predictions_stack, targets):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import GradientBoostingClassifier
    from scipy.stats import randint as sp_randint
    from sklearn.metrics.scorer import neg_log_loss_scorer
    X = predictions_stack[:, :, 1]
    X = X.T
    y = targets

    param_dist = {"max_depth": sp_randint(1, 6),
                  "n_estimators": sp_randint(1, 50),
                  "max_features": sp_randint(1, 4),
                  'min_samples_split': sp_randint(2, 3)}

    clf = GradientBoostingClassifier()

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, random_state=0)
    rscv = RandomizedSearchCV(clf, param_distributions=param_dist, cv=skf, n_jobs=4, n_iter=100,
                              scoring=neg_log_loss_scorer, verbose=1)

    rscv.fit(X, y)

    cv_result = {
        'model': rscv,
        # 'weights': rscv.best_estimator_.feature_importances_,
        'training_loss': np.mean(rscv.cv_results_['mean_train_score']),
        'validation_loss': np.mean(rscv.cv_results_['mean_test_score']),
        'configs': [CONFIGS]
    }

    return cv_result


def weighted_average(predictions, weights):
    """
    Arithmetic average

    :param predictions: (config_name -> (pid -> prediction) )
    :param weights: (config_name -> (weight) )
    :return  predictions as dict: (pid -> prediction)
    """
    weighted_predictions = {}
    for config_name, config_predictions in predictions.iteritems():
        for pid, patient_prediction in config_predictions.iteritems():
            weighted_prediction = patient_prediction * weights[config_name]
            if pid in weighted_predictions:
                weighted_predictions[pid] += weighted_prediction
            else:
                weighted_predictions[pid] = weighted_prediction

    return collections.OrderedDict(sorted(weighted_predictions.items()))


def compare_test_performance_ind_vs_ensemble(test_set_predictions):
    individual_performance = {config: calc_test_performance(config, pred_test) for config, pred_test in
                              test_set_predictions.iteritems()}
    for config, performance in individual_performance.iteritems():
        print 'Logloss of config {} is {} on test set'.format(config, performance)
    loss = evaluate_submission.leaderboard_performance(
        utils_ensemble.get_destination_path('test_set_predictions.csv', expid))
    print('Ensemble test set performance as it would be on the leaderboard: ')
    print(loss)


def calc_test_performance(config_name, predictions):
    config_name = config_name.replace('/', '')
    tmp_submission_file = '/tmp/submission_test_predictions_{}.csv'.format(config_name)
    utils_lung.write_submission(predictions, tmp_submission_file)
    loss = evaluate_submission.leaderboard_performance(tmp_submission_file)
    os.remove(tmp_submission_file)
    return loss


if __name__ == '__main__':
    print 'Starting ensembling with configs', CONFIGS
    linear_stacking()
    print 'Job done'
