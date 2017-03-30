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

CONFIGS = ['dsb_a04_c3ns2_mse_s5_p8a1', 'dsb_a07_c3ns3_mse_s5_p8a1', 'dsb_a08_c3ns3_mse_s5_p8a1',
           'dsb_a_liolme16_c3_s2_p8a1', 'dsb_a_liox6_c3_s2_p8a1', 'dsb_a_liox7_c3_s2_p8a1']

expid = utils.generate_expid('ensemble')

img_dir = '/home/adverley/Code/Projects/Kaggle/dsb3/figures/'


def ensemble():
    valid_set_predictions, valid_set_labels = load_validation_set()
    analyse_predictions(valid_set_predictions, valid_set_labels)

    weights = optimize_weights(valid_set_predictions, valid_set_labels)  # (config_name -> (weight) )

    y_valid_pred = weighted_average(valid_set_predictions, weights)

    test_set_predictions = {config: get_predictions_of_config(config, 'test') for config in CONFIGS}
    y_test_pred = weighted_average(test_set_predictions, weights)
    utils_ensemble.persist_predictions(y_test_pred, y_valid_pred, expid)
    compare_test_performance_ind_vs_ensemble(test_set_predictions)


def load_validation_set():
    valid_set_predictions = {}  # (config_name -> (pid -> prediction) )
    for config in CONFIGS:
        valid_set_predictions[config] = get_predictions_of_config(config, 'valid')
    valid_set_labels = load_validation_labels()  # (pid -> prediction)
    sanity_check(valid_set_predictions, valid_set_labels)
    return valid_set_predictions, valid_set_labels


def analyse_predictions(valid_set_predictions, labels):
    from scipy.stats import pearsonr

    print 'Correlation between predictions: '
    X = predictions_dict_to_3d_array(valid_set_predictions, labels)
    X = X[:, :, 0]

    config_names = valid_set_predictions.keys()
    amount_configs = X.shape[0]
    for config_nr in range(amount_configs):
        compare_with_nr = config_nr + 1
        while compare_with_nr < amount_configs:
            corr = pearsonr(X[config_nr, :], X[compare_with_nr, :])
            print 'Correlation between config {} and {} is {:0.2f} with p-value ({:f})' \
                .format(config_names[config_nr], config_names[compare_with_nr], corr[0], corr[1])
            compare_with_nr += 1

    corr = np.corrcoef(X)
    correlation_matrix(corr, config_names)


def correlation_matrix(corr_matrix, config_names):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(corr_matrix, interpolation="nearest", cmap=cmap)
    plt.title('Config prediction Correlation')
    labels = config_names
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    fig.colorbar(cax)

    plt.savefig(img_dir + 'correlation_between_configs.png')
    plt.close('all')


def get_predictions_of_config(config_name, which_set):
    predictions_dir = os.path.join(pathfinder.METADATA_PATH, 'model-predictions')
    exp_id = utils_ensemble.find_model_preds_expid(predictions_dir, config_name)

    output_pkl_file = predictions_dir + '/%s-%s-%s.pkl' % (config_name, exp_id, which_set)
    preds = utils.load_pkl(output_pkl_file)  # pid2prediction
    preds = collections.OrderedDict(sorted(preds.items()))
    print 'Passing predicions from {}'.format(output_pkl_file)
    return preds


def load_validation_labels():
    train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
    valid_pids = train_valid_ids['validation']
    id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)

    labels = {pid: id2label[pid] for pid in sorted(valid_pids)}
    return collections.OrderedDict(sorted(labels.items()))


def sanity_check(valid_set_predictions, valid_set_labels):
    for config in CONFIGS:
        # Check whether all configs exist
        # utils.find_model_metadata(pathfinder.METADATA_PATH, config)

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

    analyse_cv_result(cv_results['optimal_linear_weights'], 'optimal_linear_weights')
    analyse_cv_result(cv_results['equal_weights'], 'equal_weights')

    weights = optimal_linear_weights(X, np.array(utils_ensemble.one_hot(y)))

    print 'Optimal weights'
    config2weights = {}
    for model_nr in range(len(predictions.keys())):
        config = predictions.keys()[model_nr]
        print 'Weight for config {} is {}'.format(config, weights[model_nr])
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
            'configs': config_names
        })

    return cv_result


def analyse_cv_result(cv_result, ensemble_method_name):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    # WEIGHT HISTOGRAM
    weights = np.array([cv['weights'] for cv in cv_result])
    print 'weight per config across folds: ', weights
    for w in range(weights.shape[1]):
        weight = weights[:, w]
        plt.hist(weight, bins=np.linspace(0, 1, 10), facecolor=colors[w % len(colors)], alpha=0.5,
                 label='config {}'.format(cv_result[0]['configs'][w]))

    plt.title('Weight histogram of configs during CV')
    plt.legend(loc='upper right')
    plt.savefig(img_dir + 'ensemble_{}_weight_histograms.png'.format(ensemble_method_name))
    plt.clf()
    plt.close('all')

    # PERFORMANCE COMPARISON ACROSS FOLDS
    losses = np.array([cv['validation_loss'] for cv in cv_result])
    print 'Validation set losses across folds: ', losses
    print 'stats of ', ensemble_method_name, scipy.stats.describe(losses)
    with open(img_dir + 'ensemble_{}_cv_performance.txt'.format(ensemble_method_name), 'w') as f:
        f.write('stats: ')
        f.write(str(scipy.stats.describe(losses)))


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
    :return: predictions as numpy array with shape [num_configs x num_patients x 2]
    """
    n_models = len(predictions.keys())
    n_patients = len(labels)
    predictions_stack = np.zeros((n_models, n_patients, 2))  # num_configs x num_patients x 2 categories
    for model_nr in range(n_models):
        config = predictions.keys()[model_nr]
        for patient_nr, patient_id in enumerate(labels.keys()):
            predictions_stack[model_nr, patient_nr, 0] = 1.0 - predictions[config][patient_id]
            predictions_stack[model_nr, patient_nr, 1] = predictions[config][patient_id]
    return predictions_stack


def simple_average(predictions_stack, targets):
    amount_of_configs = predictions_stack.shape[0]
    equal_weight = 1.0 / amount_of_configs

    weights = [equal_weight for _ in range(amount_of_configs)]
    return weights


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
    tmp_submission_file = '/tmp/submission_test_predictions_{}.csv'.format(config_name)
    utils_lung.write_submission(predictions, tmp_submission_file)
    loss = evaluate_submission.leaderboard_performance(tmp_submission_file)
    os.remove(tmp_submission_file)
    return loss


if __name__ == '__main__':
    print 'Starting ensembling with configs', CONFIGS
    ensemble()
    print 'Job done'
