import collections

from theano.gof.utils import memoize

import pathfinder
import utils
import utils_lung
from ensemble import utils_ensemble
import os
import data_loading as  dl


def load_validation_data_spl(configs):
    valid_set_predictions = collections.OrderedDict()  # (config_name -> (pid -> prediction) )
    for config in configs:
        valid_set_predictions[config] = get_predictions_of_config(config, 'valid')
    valid_set_labels = dl.load_validation_labels()  # (pid -> prediction)
    dl.sanity_check(configs, valid_set_predictions, valid_set_labels)
    return valid_set_predictions, valid_set_labels


def load_test_data_spl(configs):
    predictions = collections.OrderedDict()  # (config_name -> (pid -> prediction) )
    for config in configs:
        predictions[config] = get_predictions_of_config(config, 'test_spl')
    return predictions, None


def load_test_data_all(configs):
    predictions = collections.OrderedDict()  # (config_name -> (pid -> prediction) )
    for config in configs:
        predictions[config] = get_predictions_of_config(config, 'test_all')
    return predictions, None


@memoize
def get_predictions_of_config(config_name, which_set):
    assert which_set in ('valid', 'test_spl', 'test_all')
    if which_set == 'test_all':
        config_name = config_name.replace('_spl', '_all')
    predictions_dir = os.path.join(pathfinder.METADATA_PATH, 'model-predictions')
    exp_id = utils_ensemble.find_model_preds_expid(predictions_dir, config_name)

    output_pkl_file = predictions_dir + '/%s-%s-%s.pkl' % (config_name, exp_id, which_set)
    preds = utils.load_pkl(output_pkl_file)  # pid2prediction
    preds = collections.OrderedDict(sorted(preds.items()))
    return preds


def sanity_check(configs, valid_set_predictions, valid_set_labels):
    for config in configs:
        # Check whether all these configs contain all the predictions
        if valid_set_predictions[config].viewkeys() != valid_set_labels.viewkeys():
            raise ValueError(
                'the validation set predictions does not contain the same pids as the validation set labels')

    pass
