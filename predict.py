"""Script for generating predictions for a given trained model.

The script loads the specified configuration file. All parameters are defined
in that file.

Usage:
> python predict.py -c CONFIG_NAME
"""
import argparse
import cPickle as pickle
import csv
import itertools
import string
import time

import datetime
from functools import partial
from itertools import izip

import lasagne
import numpy as np
import theano
import theano.tensor as T
from interfaces.data_loader import IDS

from theano_utils import theano_printer
import os
from utils import buffering
import utils
import math

from utils.configuration import config, set_configuration
from utils.paths import MODEL_PATH, MODEL_PREDICTIONS_PATH, SUBMISSION_PATH

def predict_model(expid, mfile=None):
    metadata_path = MODEL_PATH + "%s.pkl" % (expid if not mfile else mfile)
    prediction_path = MODEL_PREDICTIONS_PATH + "%s.pkl" % expid
    submission_path = SUBMISSION_PATH + "%s.csv" % expid

    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"

    print "Using"
    print "  %s" % metadata_path
    print "To generate"
    print "  %s" % prediction_path

    print "Build model"
    interface_layers = config.build_model()

    output_layers = interface_layers["outputs"]
    input_layers = interface_layers["inputs"]
    top_layer = lasagne.layers.MergeLayer(
        incomings=output_layers.values()
    )
    all_layers = lasagne.layers.get_all_layers(top_layer)
    all_params = lasagne.layers.get_all_params(top_layer, trainable=True)

    num_params = sum([np.prod(p.get_value().shape) for p in all_params])

    print string.ljust("  layer output shapes:",34),
    print string.ljust("#params:",10),
    print string.ljust("#data:",10),
    print "output shape:"
    for layer in all_layers[:-1]:
        name = string.ljust(layer.__class__.__name__, 30)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(int(num_param).__str__(), 10)
        num_size = string.ljust(np.prod(layer.output_shape[1:]).__str__(), 10)
        print "    %s %s %s %s" % (name,  num_param, num_size, layer.output_shape)
    print "  number of parameters: %d" % num_params

    xs_shared = {
        key: lasagne.utils.shared_empty(dim=len(l_in.output_shape), dtype='float32')
        for (key, l_in) in input_layers.iteritems()
    }

    idx = T.lscalar('idx')

    givens = dict()

    for (key, l_in) in input_layers.iteritems():
        givens[l_in.input_var] = xs_shared[key][idx*config.batch_size:(idx+1)*config.batch_size]

    network_outputs = [
        lasagne.layers.helper.get_output(network_output_layer, deterministic=True)
        for network_output_layer in output_layers.values()
    ]

    print "Compiling..."
    iter_test = theano.function([idx],
                                 network_outputs + theano_printer.get_the_stuff_to_print(),
                                 givens=givens, on_unused_input="ignore",
                                 # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                                 )

    required_input = {
        key: l_in.output_shape
        for (key, l_in) in input_layers.iteritems()
    }

    print "Preparing dataloaders"
    config.test_data.prepare()
    chunk_size = config.batches_per_chunk * config.batch_size

    test_data_generator = buffering.buffered_gen_threaded(
        config.test_data.generate_batch(
            chunk_size = chunk_size,
            required_input = required_input,
            required_output = {},
        )
    )


    print "Load model parameters for resuming"
    resume_metadata = np.load(metadata_path)
    lasagne.layers.set_all_param_values(top_layer, resume_metadata['param_values'])

    chunks_test_idcs = itertools.count(0)
    num_chunks_test = math.ceil(1.0 * config.test_data.epochs * config.test_data.number_of_samples / (config.batch_size * config.batches_per_chunk))

    start_time,prev_time = None,None
    all_predictions = dict()


    print "Loading first chunks"
    for e, test_data in izip(chunks_test_idcs, test_data_generator):

        if start_time is None:
            start_time = time.time()
            prev_time = start_time
        print

        print "Chunk %d/%d" % (e + 1, num_chunks_test)
        print "=============="

        if config.dump_network_loaded_data:
            pickle.dump(test_data, open("data_loader_dump_test_%d.pkl" % e, "wb"))

        for key in xs_shared:
            xs_shared[key].set_value(test_data["input"][key])

        sample_ids = test_data[IDS]

        for b in xrange(config.batches_per_chunk):
            th_result = iter_test(b)

            predictions = th_result[:len(network_outputs)]

            for output_idx, key in enumerate(output_layers.keys()):
                for sample_idx in xrange(b*config.batch_size, (b+1)*config.batch_size):
                    prediction_pos = sample_idx % config.batch_size
                    sample_id = sample_ids[sample_idx]
                    if sample_id is not None:
                        if sample_id not in all_predictions:
                            all_predictions[sample_id] = dict()
                        if key not in all_predictions[sample_id]:
                            all_predictions[sample_id][key] = predictions[output_idx][prediction_pos]
                        else:
                            all_predictions[sample_id][key] = np.concatenate((all_predictions[sample_id][key],predictions[output_idx][prediction_pos]),axis=0)


        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        print "  %s since start (+%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        try:
            if num_chunks_test:
                est_time_left = time_since_start * (float(num_chunks_test - (e + 1)) / float(e + 1))
                eta = datetime.datetime.now() + datetime.timedelta(seconds=est_time_left)
                eta_str = eta.strftime("%c")
                print "  estimated %s to go"  % utils.hms(est_time_left)
                print "  (ETA: %s)" % eta_str
        except OverflowError:
            print "  This will take really long, like REALLY long."

        print "  %dms per testing sample" % (1000.*time_since_start / ((e+1) * config.batch_size * config.batches_per_chunk))


    with open(prediction_path, 'w') as f:
        pickle.dump({
            'metadata_path': metadata_path,
            'prediction_path': prediction_path,
            'configuration_file': config.__name__,
            'git_revision_hash': utils.get_git_revision_hash(),
            'experiment_id': expid,
            'predictions': all_predictions,
        }, f, pickle.HIGHEST_PROTOCOL)

    print "  saved to %s" % prediction_path
    print

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-m', '--metadata',
                          help='metadatafile to use',
                          required=False)

    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(args.config)
    mfile = args.metadata

    predict_model(expid, mfile)