import numpy as np
import csv
import collections
import sys
import utils_lung
import pathfinder


# Call this method to know to leaderboard_performance
def leaderboard_performance(submission_file_path):
    real = utils_lung.read_test_labels(pathfinder.TEST_LABELS_PATH)
    pred = parse_predictions(submission_file_path)

    real = collections.OrderedDict(sorted(real.iteritems()))
    pred = collections.OrderedDict(sorted(pred.iteritems()))

    check_validity(real, pred)

    return utils_lung.log_loss(real.values(), pred.values())


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


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     sys.exit("Usage: evaluate_submission.py <absolute path to csv")
    #
    # submission_path = sys.argv[1]
    submission_path = '/home/user/Downloads/submission_0.55555.csv'
    loss = leaderboard_performance(submission_path)
    print loss
