import argparse
from operator import itemgetter
from interfaces.data_loader import TEST
from utils.configuration import set_configuration, config
import utils
from utils.paths import MODEL_PREDICTIONS_PATH, SUBMISSION_PATH
import cPickle as pickle
import csv
import re

__author__ = 'jonas'

def generate_submission(expid):
    """
    Generate a submission file for this contest, using a specific model predictions path
    :param expid:
    :return:
    """
    raise NotImplementedError()

    prediction_path = MODEL_PREDICTIONS_PATH + "%s.pkl" % expid
    submission_path = SUBMISSION_PATH + "%s.csv" % expid

    print "Using"
    print "  %s" % prediction_path
    print "To generate"
    print "  %s" % submission_path


    config.test_data.prepare()
    print "Opening predictions..."
    with open(prediction_path, 'rb') as f:
        predictions = pickle.load(f)['predictions']
        l = []
        for sample_id, value_dict in predictions.iteritems():
            idx = config.test_data.indices[TEST].index(sample_id)
            label = config.test_data.names[TEST][idx]
            row = [label]
            row.extend(value_dict["predicted_class"][:])
            l.append(row)

    l = sorted(l, key=lambda item: int(re.findall(r'\d+', item[0])[0]))
    print "Writing csv file..."
    with open(submission_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')

        csv_writer.writerows([["img","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"]])
        csv_writer.writerows(l)

    return
