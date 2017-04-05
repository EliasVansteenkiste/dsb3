import collections

from theano.gof.utils import memoize

import pathfinder
import utils
import utils_lung
from ensemble import utils_ensemble
import os


# Cache variables

valid_set_predictions = collections.OrderedDict()
valid_set_labels = collections.OrderedDict()
test_set_predictions = collections.OrderedDict()
test_set_labels = collections.OrderedDict()


def load_test_set(configs):
    """

     :param configs: config names as list
     :return: [test set predictions as dict (config_name -> (pid -> prediction) ),
               test set labels as dict  (pid -> prediction)  ]
     """
    test_set_predictions = collections.OrderedDict()  # (config_name -> (pid -> prediction) )
    for config in configs:
        test_set_predictions[config] = get_predictions_of_config(config, 'test')
    test_set_labels = load_test_labels()  # (pid -> prediction)
    return test_set_predictions, test_set_labels


def load_validation_set(configs):
    """

    :param configs: config names as list
    :return: [validation set predictions as dict (config_name -> (pid -> prediction) ),
              validation set labels as dict  (pid -> prediction)  ]
    """
    valid_set_predictions = collections.OrderedDict()  # (config_name -> (pid -> prediction) )
    for config in configs:
        valid_set_predictions[config] = get_predictions_of_config(config, 'valid')
    valid_set_labels = load_validation_labels()  # (pid -> prediction)
    sanity_check(configs, valid_set_predictions, valid_set_labels)
    return valid_set_predictions, valid_set_labels

@memoize
def get_predictions_of_config(config_name, which_set):
    predictions_dir = os.path.join(pathfinder.METADATA_PATH, 'model-predictions')
    exp_id = utils_ensemble.find_model_preds_expid(predictions_dir, config_name)

    output_pkl_file = predictions_dir + '/%s-%s-%s.pkl' % (config_name, exp_id, which_set)
    preds = utils.load_pkl(output_pkl_file)  # pid2prediction
    preds = collections.OrderedDict(sorted(preds.items()))
    return preds


def load_validation_labels():
    train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
    valid_pids = train_valid_ids['validation']
    id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)

    labels = {pid: id2label[pid] for pid in sorted(valid_pids)}
    return collections.OrderedDict(sorted(labels.items()))


def load_test_labels():
    real = utils_lung.read_test_labels(pathfinder.TEST_LABELS_PATH)
    return collections.OrderedDict(sorted(real.iteritems()))


def sanity_check(configs, valid_set_predictions, valid_set_labels):
    for config in configs:
        # Check whether all these configs contain all the predictions
        if valid_set_predictions[config].viewkeys() != valid_set_labels.viewkeys():
            raise ValueError(
                'the validation set predictions does not contain the same pids as the validation set labels')

    pass
