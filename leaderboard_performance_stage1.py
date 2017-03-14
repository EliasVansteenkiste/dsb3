import numpy as np
import csv
import collections


# Call this method to know to leaderboard_performance
# TODO put this as runtime method of the script
def leaderboard_performance(submission_file_path):
    real = get_stage_1_test_labels()
    pred = parse_predictions(submission_file_path)

    real = collections.OrderedDict(sorted(real.iteritems()))
    pred = collections.OrderedDict(sorted(pred.iteritems()))

    check_validity(real, pred)

    ll = log_loss(real.values(), pred.values())
    print('Leaderboard performance of submission "{}" is {}'.format(submission_file_path, ll))


def get_stage_1_test_labels():
    """
        Get test labels as inferred by Jeroen
    :return: test labels stored in a dict (patient id -> cancer)
    """
    # TODO find out where test labels of jeroen are
    test_labels = {}
    return test_labels


def parse_predictions(submission_file_path):
    pred = {}
    with open(submission_file_path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            pred[row['id']] = float(row['cancer'])
    return pred


def check_validity(real, pred):
    if len(real) != len(pred):
        raise ValueError(
            'The amount of test set labels (={}) does not match with the amount of predictions (={})'.format(len(real),
                                                                                                             len(pred)))

    if len(real.viewkeys() & pred.viewkeys()) != len(real):
        raise ValueError(
            'The patients in the test set does not match with the patients in the predictions'
        )

    if real.viewkeys() != pred.viewkeys():
        raise ValueError(
            'The patients in the test set does not match with the patients in the predictions'
        )


def log_loss(y_real, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_real = np.array(y_real)
    losses = y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred)
    return - np.average(losses)
