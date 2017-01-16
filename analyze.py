"""
For running analysis on the outputs of your model (see also application/analyze.py)

Usage:
python analyze.py myconfigfile [-f analysis_function_to_run]
"""
from application.analyze import analyze
import argparse
from functools import partial
from itertools import izip
import cPickle as pickle
import string
import datetime
import itertools
import lasagne
import time
from interfaces.data_loader import VALIDATION
from interfaces.data_loader import TRAINING
from interfaces.data_loader import IDS
from utils.log import print_to_file

from utils.configuration import set_configuration, config
import utils
from utils import LOGS_PATH, MODEL_PATH, ANALYSIS_PATH
import theano
import numpy as np
import theano.tensor as T
from theano_utils import theano_printer
import os
from utils import buffering
from utils.timer import Timer
import os

import warnings
warnings.simplefilter("error")
import sys
sys.setrecursionlimit(10000)

def analyze_model(expid, mfile=None):
    metadata_path = MODEL_PATH + "%s.pkl" % (expid if not mfile else mfile)
    analysis_path = ANALYSIS_PATH + "%s/" % expid
    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)

    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"

    print "Using"
    print "  %s" % metadata_path
    print "To generate"
    print "  %s" % analysis_path

    interface_layers = config.build_model()

    output_layers = interface_layers["outputs"]
    input_layers = interface_layers["inputs"]
    top_layer = lasagne.layers.MergeLayer(
        incomings=output_layers.values()
    )
    all_layers = lasagne.layers.get_all_layers(top_layer)
    all_params = lasagne.layers.get_all_params(top_layer, trainable=True)

    if "cutoff_gradients" in interface_layers:
        submodel_params = [param for value in interface_layers["cutoff_gradients"] for param in lasagne.layers.get_all_params(value)]
        all_params = [p for p in all_params if p not in submodel_params]

    if "pretrained" in interface_layers:
        for config_name, layers_dict in interface_layers["pretrained"].iteritems():
            pretrained_metadata_path = MODEL_PATH + "%s.pkl" % config_name.split('.')[1]
            pretrained_resume_metadata = np.load(pretrained_metadata_path)
            pretrained_top_layer = lasagne.layers.MergeLayer(
                incomings = layers_dict.values()
            )
            lasagne.layers.set_all_param_values(pretrained_top_layer, pretrained_resume_metadata['param_values'])

    num_params = sum([np.prod(p.get_value().shape) for p in all_params])

    print string.ljust("  layer output shapes:",34),
    print string.ljust("#params:",10),
    print string.ljust("#data:",10),
    print "output shape:"
    def comma_seperator(v):
        return '{:,.0f}'.format(v)

    for layer in all_layers[:-1]:
        name = string.ljust(layer.__class__.__name__, 30)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(comma_seperator(num_param), 10)
        num_size = string.ljust(comma_seperator(np.prod(layer.output_shape[1:])), 10)
        print "    %s %s %s %s" % (name,  num_param, num_size, layer.output_shape)
    print "  number of parameters:", comma_seperator(num_params)

    objectives = config.build_objectives(interface_layers)

    xs_shared = {
        key: lasagne.utils.shared_empty(dim=len(l_in.output_shape), dtype='float32')
        for (key, l_in) in input_layers.iteritems()
    }

    ys_shared = {
        key: lasagne.utils.shared_empty(dim=target_var.ndim, dtype=target_var.dtype)
        for (_,ob) in itertools.chain(objectives["train"].iteritems(), objectives["validate"].iteritems())
        for (key, target_var) in ob.target_vars.iteritems()
    }

    idx = T.lscalar('idx')

    givens = dict()

    for (_,ob) in itertools.chain(objectives["train"].iteritems(), objectives["validate"].iteritems()):
        for (key, target_var) in ob.target_vars.iteritems():
            givens[target_var] = ys_shared[key][idx*config.batch_size : (idx+1)*config.batch_size]

    for (key, l_in) in input_layers.iteritems():
        givens[l_in.input_var] = xs_shared[key][idx*config.batch_size:(idx+1)*config.batch_size]

    print "Compiling..."
    outputs = [lasagne.layers.helper.get_output(interface, deterministic=True) for interface in interface_layers["outputs"].values()]


    iter_validate = theano.function([idx],
                                    outputs + theano_printer.get_the_stuff_to_print(),
                                    givens=givens, on_unused_input="ignore")

    required_input = {
        key: l_in.output_shape
        for (key, l_in) in input_layers.iteritems()
    }
    required_output = {
        key: None  # size is not needed
        for (_,ob) in itertools.chain(objectives["train"].iteritems(), objectives["validate"].iteritems())
        for (key, target_var) in ob.target_vars.iteritems()
    }

    print "Preparing dataloaders"
    config.training_data.prepare()
    for validation_data in config.validation_data.values():
        validation_data.prepare()

    chunk_size = config.batches_per_chunk * config.batch_size

    training_data_generator = buffering.buffered_gen_threaded(
        config.training_data.generate_batch(
            chunk_size = chunk_size,
            required_input = required_input,
            required_output = required_output,
        )
    )


    print "Will train for %s epochs" % config.training_data.epochs

    if os.path.isfile(metadata_path):
        print "Load model parameters for resuming"
        resume_metadata = np.load(metadata_path)
        lasagne.layers.set_all_param_values(top_layer, resume_metadata['param_values'])
    else:
        raise "No previous parameters found!"

    start_time,prev_time = None,None
    print "Loading first chunks"
    data_load_time = Timer()
    gpu_time = Timer()

    data_load_time.start()
    for dataset_name, dataset_generator in config.validation_data.iteritems():
        data_load_time.stop()
        if start_time is None:
            start_time = time.time()
            prev_time = start_time

        validation_chunk_generator = dataset_generator.generate_batch(
                chunk_size = chunk_size,
                required_input = required_input,
                required_output = required_output,
            )

        print "  %s (%d/%d samples)" % (dataset_name, dataset_generator.number_of_used_samples, dataset_generator.number_of_samples)
        print "  -----------------------"

        data_load_time.start()
        for validation_data in buffering.buffered_gen_threaded(validation_chunk_generator):
            data_load_time.stop()
            num_batches_chunk_eval = config.batches_per_chunk

            for key in xs_shared:
                xs_shared[key].set_value(validation_data["input"][key])

            for key in ys_shared:
                ys_shared[key].set_value(validation_data["output"][key])

            idx = 0
            for b in xrange(num_batches_chunk_eval):
                gpu_time.start()
                th_result = iter_validate(b)
                gpu_time.stop()

                for idx_ex in xrange(config.batch_size):
                    kwargs = {}
                    for key in xs_shared.keys():
                        kwargs[key] = validation_data["input"][key][idx+idx_ex]

                    for key in ys_shared.keys():
                        kwargs[key] = validation_data["output"][key][idx+idx_ex]

                    for index, key in enumerate(interface_layers["outputs"].keys()):
                        kwargs[key] = th_result[index][idx_ex]

                    id = validation_data[IDS][idx+idx_ex]
                    if id is not None:
                        analyze(id=id, analysis_path=analysis_path, **kwargs)

                idx += config.batch_size

            data_load_time.start()
        data_load_time.stop()
        print


        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        print "  %s since start (+%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  (%s waiting on gpu vs %s waiting for data)" % (gpu_time, data_load_time)
        gpu_time.reset()
        data_load_time.reset()
        data_load_time.start()

    return





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)
    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(args.config)

    log_file = LOGS_PATH + "%s_analysis.log" % expid
    with print_to_file(log_file):

        print "Running configuration:", config.__name__
        print "Current git version:", utils.get_git_revision_hash()

        analyze_model(expid)
        print "log saved to '%s'" % log_file