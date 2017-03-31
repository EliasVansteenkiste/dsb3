import os

from os import path
import theano.tensor as T
import theano
import numpy as np
import pathfinder
import utils
import utils_lung
import glob


def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.sum(t * T.log(y)) / y.shape[0].astype(theano.config.floatX)
    return loss


def log_losses(y, t, eps=1e-15):
    """
    cross entropy loss per example, summed over classes
    """
    y = T.clip(y, eps, 1 - eps)
    losses = -T.sum(t * T.log(y), axis=1)
    return losses


def one_hot(vec, m=None):
    if m is None:
        m = int(np.max(vec)) + 1

    return np.eye(m)[vec]


def persist_predictions(y_test_pred, y_valid_pred, expid):
    utils.save_pkl(y_valid_pred, get_destination_path('validation_set_predictions.pkl', expid))
    print 'Pickled ensemble predictions on validation set ({})'.format(
        get_destination_path('validation_set_predictions.pkl', expid))

    utils.save_pkl(y_test_pred, get_destination_path('test_set_predictions.pkl', expid))
    print 'Pickled ensemble predictions on test set ({})'.format(
        get_destination_path('test_set_predictions.pkl', expid))

    utils_lung.write_submission(y_test_pred, get_destination_path('test_set_predictions.csv', expid))
    print 'Saved ensemble predictions into csv file ({})'.format(
        get_destination_path('test_set_predictions.csv', expid))


def get_destination_path(filename, expid):
    ensemble_predictions_dir = os.path.join(pathfinder.METADATA_PATH, 'model-predictions', 'ensemble')
    utils.auto_make_dir(ensemble_predictions_dir)

    destination_folder = path.join(ensemble_predictions_dir, expid)
    utils.auto_make_dir(destination_folder)
    destination_path = path.join(destination_folder, filename)
    return destination_path


def find_model_preds_expid(preds_dir, config_name):
    if config_name.find('/') != -1:
        user_folder = config_name[0:config_name.index('/')]
        config_name = config_name[config_name.index('/') + 1:]
        preds_dir += '/' + user_folder

    paths = glob.glob(preds_dir + '/%s-*' % config_name)
    # black magic that simply extracts the timestamp
    exp_ids = list(set(
        [path[path.find(config_name) + len(config_name) + 1:path.find(config_name) + len(config_name) + 16] for path in
         paths]))
    if not paths:
        raise ValueError('No prediction files for config %s' % (config_name))
    elif len(exp_ids) > 1:
        print '\nWARNING: Multiple prediction files for config %s' % config_name
        print 'Taking the most recent predictions for this config...\n'
        return exp_ids[len(exp_ids) - 1]
    return exp_ids[0]
