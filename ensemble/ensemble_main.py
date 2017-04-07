import utils
import pathfinder
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

EV_CONFIGS = ['eavsteen/' + config for config in ['dsb_a_eliasq1_mal2_s5_p8a1', 'dsb_a_eliasq10_mal2_s5_p8a1']]

CONFIGS = ['dsb_a04_c3ns2_mse_s5_p8a1', 'dsb_a07_c3ns3_mse_s5_p8a1', 'dsb_a08_c3ns3_mse_s5_p8a1',
           'dsb_a11_m1zm_s5_p8a1', 'dsb_af25lmeaapm_mal2_s5_p8a1', 'dsb_a_liolme16_c3_s2_p8a1',
           'dsb_a_liolme32_c3_s5_p8a1', 'dsb_a_liox10_c3_s2_p8a1', 'dsb_a_liox11_c3_s5_p8a1', 'dsb_a_liox12_c3_s2_p8a1',
           'dsb_af25lmelr10-2_mal2_s5_p8a1', 'dsb_af25lmelr10-1_mal2_s5_p8a1', 'dsb_a_eliasz1_c3_s5_p8a1',
           'dsb_a_liox13_c3_s2_p8a1', 'dsb_a_liox14_c3_s2_p8a1', 'dsb_a_liox15_c3_s2_p8a1', 'dsb_a_liox6_c3_s2_p8a1',
           'dsb_a_liox7_c3_s2_p8a1', 'dsb_a_liox8_c3_s2_p8a1', 'dsb_a_liolunalme16_c3_s2_p8a1',
           'dsb_a_lionoclip_c3_s5_p8a1', 'dsb_a_liomse_c3_s5_p8a1', 'dsb_af25lmeo0_s5_p8a1',
           'dsb_a_liomseresume_c3_s5_p8a1', 'dsb_af25lmelr10-3_mal2_s5_p8a1', 'dsb_a_liomix_c3_s5_p8a1',
           'dsb_a_liomselunaresume_c3_s5_p8a1']

GOOD_CONFIGS = ['fgodin/dsb_af25lme_mal2_s5_p8a1', 'dsb_af25lmeaapm_mal2_s5_p8a1', 'dsb_a_liolme16_c3_s2_p8a1',
                'dsb_a_liolme32_c3_s5_p8a1', 'dsb_af25lmelr10-2_mal2_s5_p8a1', 'dsb_a_liox13_c3_s2_p8a1',
                'dsb_af25lmeo0_s5_p8a1', 'dsb_af25lmelr10-3_mal2_s5_p8a1', 'eavsteen/dsb_a_eliasq1_mal2_s5_p8a1']

CONFIGS = FG_CONFIGS + CONFIGS + EV_CONFIGS

OUTLIER_THRESHOLD = 0.10  # Disagreement threshold (%)
DO_CV = False
VERBOSE = False


def pruning_ensemble(configs, with_majority_vote):
    """
    Take models trained on all the data. Do a cross validation to get a ranking between the models. Choose the top N models.
    Merge these top N model into an equally weighted model.
    """
    name = 'pruning_ensembling_{}'.format('with_majority_vote' if with_majority_vote else 'without_majority_vote')

    ensemble_info = {}
    ensemble_info['Name'] = name
    expid = utils.generate_expid(name)

    X_valid, y_valid = load_data(configs, 'validation')
    anal.analyse_predictions(X_valid, y_valid)

    cv = do_cross_validation(X_valid, y_valid, configs, em.optimal_linear_weights)
    if DO_CV:
        anal.analyse_cv_result(cv, 'linear optimal weight')
        anal.analyse_cv_result(do_cross_validation(X_valid, y_valid, configs, em.equal_weights), 'equal weight')

    configs_to_use = prune_configs(configs, cv)
    ensemble_info['final ensemble will use configs'] = configs_to_use

    X_valid, y_valid = load_data(configs_to_use, 'validation')
    ensemble_model = em.WeightedEnsemble(configs_to_use, optimization_method=em.equal_weights)
    ensemble_model.train(X_valid, y_valid)
    ensemble_info['Ensemble training error'] = ensemble_model.training_error
    ensemble_info['Ensemble model weights'] = ensemble_model.print_weights()
    ensemble_info['ensemble_model'] = ensemble_model

    X_test, y_test = load_data(configs_to_use, 'test')
    test_pids = y_test.keys()

    y_test_pred = {}

    for pid in test_pids:
        test_sample = filter_set(X_test, pid, configs_to_use)
        ensemble_pred = ensemble_model.predict_one_sample(test_sample)
        y_test_pred[pid] = majority_vote_rensemble_prediction(X_test, ensemble_pred,
                                                              pid) if with_majority_vote else ensemble_pred

    ensemble_info['out_of_sample_error'] = evaluate_test_set_performance(y_test, y_test_pred)
    ensemble_info['y_test_pred'] = y_test_pred
    utils_ensemble.persist_test_set_predictions(expid, y_test_pred)

    return ensemble_info


def optimal_linear_ensembling(configs, with_majority_vote):
    """
    Take models trained on training data. Optimise the hell out of it using the validation data.
    This is to protect against overfitted or very bad models.
    """
    name = 'optimal_linear_ensemble_{}'.format('with_majority_vote' if with_majority_vote else 'without_majority_vote')

    ensemble_info = {}
    ensemble_info['Name'] = name
    expid = utils.generate_expid(name)

    X_valid, y_valid = load_data(configs, 'validation')
    anal.analyse_predictions(X_valid, y_valid)

    cv = do_cross_validation(X_valid, y_valid, configs, em.optimal_linear_weights)
    if DO_CV:
        anal.analyse_cv_result(cv, 'linear optimal weight')
        anal.analyse_cv_result(do_cross_validation(X_valid, y_valid, configs, em.equal_weights), 'equal weight')

    ensemble_model = em.WeightedEnsemble(configs, optimization_method=em.optimal_linear_weights)
    ensemble_model.train(X_valid, y_valid)
    ensemble_info['ensemble training error'] = ensemble_model.training_error
    ensemble_info['Ensemble model weights'] = ensemble_model.print_weights()

    X_test, y_test = load_data(configs, 'test')
    test_pids = y_test.keys()

    y_test_pred = {}

    for pid in test_pids:
        test_sample = filter_set(X_test, pid, configs)
        ensemble_pred = ensemble_model.predict_one_sample(test_sample)
        y_test_pred[pid] = majority_vote_rensemble_prediction(X_test, ensemble_pred,
                                                              pid) if with_majority_vote else ensemble_pred

    ensemble_info['out_of_sample_error'] = evaluate_test_set_performance(y_test, y_test_pred)
    ensemble_info['y_test_pred'] = y_test_pred
    utils_ensemble.persist_test_set_predictions(expid, y_test_pred)
    return ensemble_info


def cv_averaged_weight_ensembling(configs, with_majority_vote):
    """
    Average the weights of a 10 SKF CV together with a full retrain of the weights to get the final ensemble weights.
    """
    name = 'cv_averaged_weight_ensembling_{}'.format(
        'with_majority_vote' if with_majority_vote else 'without_majority_vote')

    ensemble_info = {}
    ensemble_info['Name'] = name
    expid = utils.generate_expid(name)

    X_valid, y_valid = load_data(configs, 'validation')
    anal.analyse_predictions(X_valid, y_valid)

    cv = do_cross_validation(X_valid, y_valid, configs, em.optimal_linear_weights)

    ensemble_model = em.WeightedEnsemble(configs, optimization_method=em.optimal_linear_weights)
    ensemble_model.train(X_valid, y_valid)

    ensemble_model = average_weights_with_cv(ensemble_model, cv)
    ensemble_info['Ensemble model weights'] = ensemble_model.print_weights()

    X_test, y_test = load_data(configs, 'test')
    test_pids = y_test.keys()

    y_test_pred = {}

    for pid in test_pids:
        test_sample = filter_set(X_test, pid, configs)
        ensemble_pred = ensemble_model.predict_one_sample(test_sample)
        y_test_pred[pid] = majority_vote_rensemble_prediction(X_test, ensemble_pred,
                                                              pid) if with_majority_vote else ensemble_pred

    ensemble_info['out_of_sample_error'] = evaluate_test_set_performance(y_test, y_test_pred)
    ensemble_info['y_test_pred'] = y_test_pred
    utils_ensemble.persist_test_set_predictions(expid, y_test_pred)
    return ensemble_info


def average_weights_with_cv(ensemble_model, cv_result):
    config_weights = dict(ensemble_model.weights)
    N = 1
    for cv in cv_result:
        N += 1
        weights = cv['weights']
        config_names = np.array(cv['configs'])

        for config_nr in range(len(config_names)):
            config_name = config_names[config_nr]
            weight = weights[config_nr]

            config_weights[config_name] = config_weights[config_name] + 1.0 / N * (weight - config_weights[config_name])

    new_ensemble_model = em.WeightedEnsemble(ensemble_model.models, optimization_method=em.linear_optimal_ensemble)
    new_ensemble_model.weights = config_weights
    return new_ensemble_model


def prune_configs(configs_used, cv_result, prune_percent=0.5):
    # prune if a config was used less than prune_percent of the time
    config_usage_count = {config_name: 0.0 for config_name in configs_used}

    for cv in cv_result:
        weights = cv['weights']
        config_names = np.array(cv['configs'])

        used_configs = config_names[np.invert((np.isclose(weights, np.zeros_like(weights))))]
        for used_config in used_configs:
            config_usage_count[used_config] += 0.1

    return [config for config in configs_used if config_usage_count[config] >= prune_percent]


def evaluate_test_set_performance(y_test, y_test_pred):
    # TODO implement this in case we first reserve a part of the validation set as final test set to use internally
    test_logloss = utils_lung.evaluate_log_loss(y_test_pred, y_test)
    return test_logloss


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


def print_individual_configs_test_set_performance(configs):
    test_set_predictions = {config: data_loading.get_predictions_of_config(config, 'test') for config in configs}
    individual_performance = {config: calc_test_performance(config, pred_test) for config, pred_test in
                              test_set_predictions.iteritems()}
    for config, performance in individual_performance.iteritems():
        print 'Logloss of config {} is {} on test set'.format(config, performance)


def print_individual_configs_validation_set_performance(configs):
    valid_set_predictions = {config: data_loading.get_predictions_of_config(config, 'valid') for config in configs}
    valid_labels = data_loading.load_validation_labels()
    for config, valid_preds in valid_set_predictions.iteritems():
        print 'Logloss of config {} is {} on validation set'.format(config, utils_lung.evaluate_log_loss(valid_preds,
                                                                                                         valid_labels))


def print_ensemble_result(result):
    for k, v in result.iteritems():
        if k != 'y_test_pred':
            print k, ': ', v

    print '\n'


print 'Starting ensemble procedure with {} configs'.format(len(CONFIGS))
ensemble_strategies = []
print '\n--------------------'
print 'INDIVIDUAL MODEL PERFORMANCE'
print '--------------------\n'
print_individual_configs_validation_set_performance(CONFIGS)
print_individual_configs_test_set_performance(CONFIGS)

print '\n--------------------'
print 'PRUNING ENSEMBLE'
print '--------------------\n'
info = pruning_ensemble(CONFIGS, with_majority_vote=False)
print_ensemble_result(info)
ensemble_strategies.append(info)

info = pruning_ensemble(CONFIGS, with_majority_vote=True)
print_ensemble_result(info)
ensemble_strategies.append(info)

print '\n--------------------'
print 'optimal_linear_ensembling'
print '--------------------\n'
info = optimal_linear_ensembling(CONFIGS, with_majority_vote=False)
print_ensemble_result(info)
ensemble_strategies.append(info)

info = optimal_linear_ensembling(CONFIGS, with_majority_vote=True)
print_ensemble_result(info)
ensemble_strategies.append(info)

print '\n--------------------'
print 'CV AVERAGED ENSEMBLE'
print '--------------------\n'
info = cv_averaged_weight_ensembling(CONFIGS, with_majority_vote=False)
print_ensemble_result(info)
ensemble_strategies.append(info)

info = cv_averaged_weight_ensembling(CONFIGS, with_majority_vote=True)
print_ensemble_result(info)
ensemble_strategies.append(info)

# Choose two final submissions using the best ensemble
lowest_log_loss = np.inf
best_strat = None
for ensemble_strat in ensemble_strategies:
    log_loss = ensemble_strat['out_of_sample_error']
    if log_loss < lowest_log_loss:
        lowest_log_loss = log_loss
        best_strat = ensemble_strat

print 'Best ensemble strategy is {} with out-of-sample error {}'.format(best_strat['Name'], lowest_log_loss)


def defensive_ensemble(CONFIGS):
    """
    Load predictions of models trained on the split, do 10 SKF CV and make optimized weighted ensemble using the 
    models that appear in the ensemble at least 50% of the folds. 
    """
    ensembling_result = pruning_ensemble(CONFIGS, with_majority_vote=False)
    return ensembling_result['ensemble_model'], ensembling_result['y_test_pred']


def offensive_ensemble(configs_to_use):
    """
    Load predictions of models specified as argument (preferably the models used by the defensive ensemble), 
    which are trained on ALL the data and make an uniform ensemble out of it. 
    """
    pass


defensive_ensemble_model, defensive_ensemble_test_predictions = defensive_ensemble(CONFIGS)
offensive_ensemble_model, offensive_ensemble_test_predictions = offensive_ensemble(defensive_ensemble_model.models)

# TODO find this path with pathfinder
ensemble_submission_path = '/mnt/storage/metadata/dsb3/submissions/ensemble/'
utils_lung.write_submission(defensive_ensemble_test_predictions,
                            ensemble_submission_path + 'final_kaggle_submission_1.csv')
utils_lung.write_submission(offensive_ensemble_test_predictions,
                            ensemble_submission_path + 'final_kaggle_submission_2.csv')
