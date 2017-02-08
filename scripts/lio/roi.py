import argparse
import theano
import numpy as np
import theano.tensor as T
import os
import sys
from itertools import izip
import cPickle as pickle
import string
import time
import datetime
import math
import itertools
import lasagne
import time

sys.path.append(".")
from theano_utils import theano_printer
import utils
from utils import LOGS_PATH, MODEL_PATH, MODEL_PREDICTIONS_PATH
from utils.log import print_to_file
from utils import buffering
from utils.configuration import set_configuration, config, get_configuration_name
from interfaces.data_loader import VALIDATION, VALID_SAMPLES, TRAINING, IDS
from interfaces.objectives import MAXIMIZE


def extract_rois(expid):
    metadata_path = MODEL_PATH + "%s.pkl" % config.model.__name__
    assert os.path.exists(metadata_path)
    prediction_path = MODEL_PREDICTIONS_PATH + "%s.pkl" % expid

    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"

    print "Using"
    print "  %s" % metadata_path
    print "To generate"
    print "  %s" % prediction_path

    print "Build model"

    interface_layers = config.model.build_model(image_size=config.patch_shape)
    output_layers = interface_layers["outputs"]
    input_layers = interface_layers["inputs"]
    for old_key, new_key in config.replace_input_tags.items():
        input_layers[new_key] = input_layers.pop(old_key)

    # merge all output layers into a fictional dummy layer which is not actually used
    top_layer = lasagne.layers.MergeLayer(
        incomings=output_layers.values()
    )

    # get all the trainable parameters from the model
    all_layers = lasagne.layers.get_all_layers(top_layer)
    all_params = lasagne.layers.get_all_params(top_layer, trainable=True)

    # Count all the parameters we are actually optimizing, and visualize what the model looks like.
    print string.ljust("  layer output shapes:", 26),
    print string.ljust("#params:", 10),
    print string.ljust("#data:", 10),
    print "output shape:"
    def comma_seperator(v):
        return '{:,.0f}'.format(v)
    for layer in all_layers[:-1]:
        name = string.ljust(layer.__class__.__name__, 22)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(comma_seperator(num_param), 10)
        num_size = string.ljust(comma_seperator(np.prod(layer.output_shape[1:])), 10)
        print "    %s %s %s %s" % (name, num_param, num_size, layer.output_shape)
    num_params = sum([np.prod(p.get_value().shape) for p in all_params])
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

    # required_input = {
    #     key: l_in.output_shape
    #     for (key, l_in) in input_layers.iteritems()
    # }

    print "Preparing dataloaders"
    config.data_loader.prepare()
    # chunk_size = config.batch_size

    # data_generator = buffering.buffered_gen_threaded(
    #     config.data_loader.generate_batch(
    #         chunk_size=chunk_size,
    #         required_input=required_input,
    #         required_output={},
    #     )
    # )

    print "Load model parameters"
    metadata = np.load(metadata_path)
    lasagne.layers.set_all_param_values(top_layer, metadata['param_values'])

    chunks_test_idcs = itertools.count(0)
    num_chunks_test = math.ceil(1.0 * config.data_loader.epochs * config.data_loader.number_of_samples / config.batch_size)

    start_time, prev_time = None, None
    all_predictions = dict()

    for set, set_indices in config.data_loader.indices.iteritems():
        for sample_id in set_indices:

            if start_time is None:
                start_time = time.time()
                prev_time = start_time
            print "sample_id", sample_id

            data = config.data_loader.load_sample(sample_id, input_layers.keys(),{})

            patch_generator = config.patch_generator(data, output_layers["predicted_segmentation"].output_shape[1:])
            # sample_ids = data[IDS]
            # from time import time
            t0 = time.time()
            for patch in patch_generator:
                # print "xs_shared.keys()", xs_shared.keys()
                # print "patch.key()", patch.keys()
                # print 'patch["stage1:3d"].shape', patch["stage1:3d"].shape
                for key in xs_shared:
                    xs_shared[key].set_value(patch[key][None,:])

                print " patch_generator", time.time() - t0

                t0 = time.time()
                th_result = iter_test(0)
                print " iter_test", time.time()-t0

                predictions = th_result[:len(network_outputs)]
                # print "len(predictions)", len(predictions)

                pred = predictions[0][0]

                t0 = time.time()
                rois = config.extract_nodules(pred, patch)
                print " extract_nodules", time.time() - t0

                # t0 = time.time()
                if rois is not None:
                    if sample_id not in all_predictions:
                        all_predictions[sample_id] = rois
                    else:
                        all_predictions[sample_id] = np.vstack((all_predictions[sample_id], rois))
                # print "vstack", time.time() - t0

                t0 = time.time()

            print "patient", sample_id, all_predictions[sample_id].shape[0], "nodules"
            now = time.time()
            time_since_start = now - start_time
            time_since_prev = now - prev_time
            prev_time = now
            print "  %s since start (+%.2f s)" % (utils.hms(time_since_start), time_since_prev)
            # try:
            #     if num_chunks_test:
            #         est_time_left = time_since_start * (float(num_chunks_test - (e + 1)) / float(e + 1))
            #         eta = datetime.datetime.now() + datetime.timedelta(seconds=est_time_left)
            #         eta_str = eta.strftime("%c")
            #         print "  estimated %s to go" % utils.hms(est_time_left)
            #         print "  (ETA: %s)" % eta_str
            # except OverflowError:
            #     print "  This will take really long, like REALLY long."

            # print "  %dms per sample" % (
            # 1000. * time_since_start / ((e + 1) * config.batch_size))

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
    parser.add_argument("config", help='configuration to run',)
    # required = parser.add_argument_group('required arguments')
    # required.add_argument('-c', '--config',
    #                       required=True)
    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(get_configuration_name())

    log_file = LOGS_PATH + "%s-train.log" % expid
    with print_to_file(log_file):

        print "Running configuration:", config.__name__
        print "Current git version:", utils.get_git_revision_hash()

        extract_rois(expid)
        print "log saved to '%s'" % log_file