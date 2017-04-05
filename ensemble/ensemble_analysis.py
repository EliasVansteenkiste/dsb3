
import numpy as np
import scipy
import scipy.stats

import utils
import utils_lung
import utils_plots
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from ensemble import utils_ensemble

analysis_dir = '/home/adverley/Code/Projects/Kaggle/dsb3/analysis'
utils.auto_make_dir(analysis_dir)
VERBOSE = False


def analyse_cv_result(cv_result, ensemble_method_name):
    """

    :param cv_result: list [ amount_folds x {weights, training_loss, validation_loss, ...} ]
    :param ensemble_method_name: optimal linear or simple average
    """
    utils.auto_make_dir(analysis_dir + '/' + ensemble_method_name)

    histogram_of_good_weights(cv_result, ensemble_method_name)
    ranking(cv_result, ensemble_method_name)

    # PERFORMANCE COMPARISON ACROSS FOLDS
    losses = np.array([cv['validation_loss'] for cv in cv_result])
    print 'Validation set losses across folds: ', losses
    print 'stats of ', ensemble_method_name, scipy.stats.describe(losses)

    #     Persist CV result to disk
    with open(analysis_dir + '/' + ensemble_method_name + '/' + 'ensemble_{}_cv_result.txt'.format(
            ensemble_method_name), 'w') as f:
        f.write('CV result: ')
        f.write(str(cv_result))
        f.write('\nstats: ')
        f.write(str(scipy.stats.describe(losses)))

    # lets look at the relationship between the weight of the individual model and the validation loss
    relationship_config_weights_validation_losses(cv_result, ensemble_method_name)


def ranking(cv_result, ensemble_method_name):
    amount_folds = len(cv_result)
    amount_models = len(cv_result[0]['weights'])
    weights = np.zeros((amount_folds, amount_models))
    for n_fold, fold in enumerate(cv_result):
        for n_weight, weight in enumerate(fold['weights']):
            weights[n_fold, n_weight] = weight
    from scipy.stats import rankdata
    rankings = np.array([len(weight) - rankdata(weight).astype(int) for weight in weights])
    model_names = cv_result[0]['configs']
    msg = 'MODEL LEADERBOARD'
    for model_nr in range(amount_models):
        model_name = model_names[model_nr]
        model_rankings = rankings[:, model_nr]
        model_rankings_count = np.bincount(model_rankings)

        msg += '\n\nModel {} ended up'.format(model_name)
        for rank in range(len(model_rankings_count)):
            msg += '\n\t\ton the {}nd place for {} times'.format(rank + 1, model_rankings_count[rank])
    print msg
    with open(analysis_dir + '/' + ensemble_method_name + '/ranking_models.txt', 'w') as f:
        f.write(msg)


def histogram_of_good_weights(cv_result, ensemble_method_name):
    weights = np.array([cv['weights'] for cv in cv_result])
    average_importance = np.mean(weights)
    configs = cv_result[0]['configs']
    if VERBOSE:
        print 'weight per config across folds: ', weights
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for w in range(weights.shape[1]):
        weight_of_config_across_folds = np.array(weights[:, w])
        if np.any(weight_of_config_across_folds > average_importance):
            plt.hist(weight_of_config_across_folds, bins=np.linspace(0, 1, 10), facecolor=colors[w % len(colors)],
                     alpha=0.5,
                     label='config {}'.format(configs[w]))
    plt.title('Weight histogram of configs during CV')
    plt.legend(loc='upper right')
    plt.savefig(analysis_dir + '/' + ensemble_method_name + '/' + 'ensemble_{}_weight_histograms.png'.format(
        ensemble_method_name))
    plt.clf()
    plt.close('all')


def relationship_config_weights_validation_losses(cv_result, ensemble_method_name):
    from ensemble import ensemble_predictions as ens
    for model_name in ens.CONFIGS:
        weight_for_model = []
        losses = []
        for cv in cv_result:
            weight_of_config_across_folds = cv['weights'][cv['configs'].index(model_name)]
            validation_loss = cv['validation_loss']
            weight_for_model.append(weight_of_config_across_folds)
            losses.append(validation_loss)

        if VERBOSE:
            print '\ncorrelation between weight for model {} and loss on validation set: '.format(
                model_name), scipy.stats.pearsonr(weight_for_model, losses)
            print zip(weight_for_model, losses)

        plt.scatter(weight_for_model, losses)
        plt.savefig(
            analysis_dir + '/' + ensemble_method_name + '/' + '{}_correlation_weight_model_{}_and_valid_loss.png'.format(
                ensemble_method_name, model_name.replace('/', '')))
        plt.close('all')


def analyse_predictions(valid_set_predictions, labels):
    from scipy.stats import pearsonr

    if VERBOSE: print 'Correlation between predictions: '
    X = utils_ensemble.predictions_dict_to_3d_array(valid_set_predictions)
    X = X[:, :, 0]

    config_names = valid_set_predictions.keys()
    amount_configs = X.shape[0]
    for config_nr in range(amount_configs):
        compare_with_nr = config_nr + 1
        while compare_with_nr < amount_configs:
            corr = pearsonr(X[config_nr, :], X[compare_with_nr, :])
            if VERBOSE: print 'Correlation between config {} and {} is {:0.2f} with p-value ({:f})' \
                .format(config_names[config_nr], config_names[compare_with_nr], corr[0], corr[1])
            compare_with_nr += 1

    corr = np.corrcoef(X)
    correlation_matrix_plot(corr, config_names)


def correlation_matrix_plot(corr_matrix, config_names):
    fig = plt.figure(figsize=(8.0, 6.0))

    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('OrRd')
    cax = ax1.imshow(corr_matrix, interpolation="nearest", cmap=cmap)
    plt.title('Config prediction Correlation')
    labels = config_names
    plt.xticks(np.arange(len(labels)), labels, fontsize=5, rotation='vertical')
    plt.margins(1.0) # extend margins so ticks don't get clipped
    plt.subplots_adjust(bottom=0.2)

    plt.yticks(np.arange(len(labels)), labels, fontsize=5)
    fig.colorbar(cax)

    plt.savefig(analysis_dir + '/correlation_between_configs.png', dpi=300)
    plt.close('all')

def performance_across_slices(configs, valid_set_predictions, valid_set_labels, test_set_predictions, test_set_labels):
    import slice_thickness_analysis as s
    import csv

    f = open(analysis_dir + '/performance_data.csv', 'wb')
    writer = csv.writer(f)
    writer.writerow(('pid', 'config', 'y', 'y_hat', 'logloss', 'dataset', 'slice_thickness', 'slice_thickness_bin'))

    pids = []
    preds = []
    reals = []
    log_losses = []
    memberships = []
    slice_thicknesses = []

    for pid in valid_set_labels.keys():
        slice_info = s.get_slice_info_of_patient(pid)

        y = valid_set_labels[pid]
        log_loss = np.mean(
            [utils_lung.log_loss(y, y_hat) for y_hat in [valid_set_predictions[config][pid] for config in CONFIGS]])

        dataset_membership = 'validation set'
        slice_thickness = slice_info['slice_thickness']
        slice_thickness_bin = -1
        if slice_thickness < 1.25:
            slice_thickness_bin = 1
        elif slice_thickness < 1.5:
            slice_thickness_bin = 2
        elif slice_thickness < 2:
            slice_thickness_bin = 3
        elif slice_thickness == 2:
            slice_thickness_bin = 4
        else:
            slice_thickness_bin = 5

        amount_slices = slice_info['amount_slices']
        x_scale = slice_info['x_scale']
        z_scale = slice_info['z_scale']
        label = slice_info['label']

        pids.append(pid)
        # preds.append(y_hat)
        # reals.append(y)
        log_losses.append(log_loss)
        memberships.append(dataset_membership)
        slice_thicknesses.append(slice_thickness)

        for config in configs:
            writer.writerow((pid, config, y, valid_set_predictions[config][pid],
                             utils_lung.log_loss(y, valid_set_predictions[config][pid]), dataset_membership,
                             slice_thickness, slice_thickness_bin))

    for pid in test_set_labels.keys():
        slice_info = s.get_slice_info_of_patient(pid)

        y = test_set_labels[pid]
        log_loss = np.mean(
            [utils_lung.log_loss(y, y_hat) for y_hat in [test_set_predictions[config][pid] for config in CONFIGS]])

        dataset_membership = 'test set'
        slice_thickness = slice_info['slice_thickness']
        slice_thickness_bin = -1
        if slice_thickness < 1.25:
            slice_thickness_bin = 1
        elif slice_thickness < 1.5:
            slice_thickness_bin = 2
        elif slice_thickness < 2:
            slice_thickness_bin = 3
        elif slice_thickness == 2:
            slice_thickness_bin = 4
        else:
            slice_thickness_bin = 5
        amount_slices = slice_info['amount_slices']
        x_scale = slice_info['x_scale']
        z_scale = slice_info['z_scale']
        label = slice_info['label']

        pids.append(pid)
        # preds.append(y_hat)
        # reals.append(y)
        log_losses.append(log_loss)
        memberships.append(dataset_membership)
        slice_thicknesses.append(slice_thickness)

        for config in configs:
            writer.writerow((pid, config, y, test_set_predictions[config][pid],
                             utils_lung.log_loss(y, test_set_predictions[config][pid]), dataset_membership,
                             slice_thickness, slice_thickness_bin))

    import matplotlib.pyplot as plt

    # convert
    log_losses = np.array(log_losses)
    slice_thicknesses = np.array(slice_thicknesses)
    X = set(slice_thicknesses)
    thickness2performance = {}
    for slice_thickness in X:
        idx = np.array(np.where(slice_thicknesses == slice_thickness)).flatten()
        avg_ce = np.mean(log_losses[idx])
        std_ce = np.std(log_losses[idx])
        thickness2performance[slice_thickness] = avg_ce

    thickness2performance = np.collections.OrderedDict(sorted(thickness2performance.items()))

    plt.plot(thickness2performance.keys(), thickness2performance.values())
    plt.savefig(analysis_dir + '/performance_per_thickness.png')
    plt.close('all')
    f.close()
