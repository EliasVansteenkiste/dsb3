"""
Run with:
python train.py myconfigfile
"""
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
from interfaces.objectives import MAXIMIZE
from utils.log import print_to_file

from utils.configuration import set_configuration, config
import utils
from utils import LOGS_PATH, MODEL_PATH
import theano
import numpy as np
import theano.tensor as T
from theano_utils import theano_printer
import os
from utils import buffering
from utils.timer import Timer

import warnings
warnings.simplefilter("error")
import sys
sys.setrecursionlimit(10000)

def train_model(expid):
    metadata_path = MODEL_PATH + "%s.pkl" % expid

    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"

    print "Build model"
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

    print string.ljust("  layer output shapes:",26),
    print string.ljust("#params:",10),
    print string.ljust("#data:",10),
    print "output shape:"
    def comma_seperator(v):
        return '{:,.0f}'.format(v)

    for layer in all_layers[:-1]:
        name = string.ljust(layer.__class__.__name__, 22)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(comma_seperator(num_param), 10)
        num_size = string.ljust(comma_seperator(np.prod(layer.output_shape[1:])), 10)
        print "    %s %s %s %s" % (name,  num_param, num_size, layer.output_shape)
    print "  number of parameters:", comma_seperator(num_params)

    objectives = config.build_objectives(interface_layers)

    train_losses_theano      = {key:ob.get_loss()
                                for key,ob in objectives["train"].iteritems()}

    validate_losses_theano = {key:ob.get_loss(deterministic=True)
                                for key,ob in objectives["validate"].iteritems()}

    xs_shared = {
        key: lasagne.utils.shared_empty(dim=len(l_in.output_shape), dtype='float32')
        for (key, l_in) in input_layers.iteritems()
    }

    ys_shared = {
        key: lasagne.utils.shared_empty(dim=target_var.ndim, dtype=target_var.dtype)
        for (_,ob) in itertools.chain(objectives["train"].iteritems(), objectives["validate"].iteritems())
        for (key, target_var) in ob.target_vars.iteritems()
    }

    learning_rate_schedule = config.learning_rate_schedule
    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))
    idx = T.lscalar('idx')

    givens = dict()

    for (_,ob) in itertools.chain(objectives["train"].iteritems(), objectives["validate"].iteritems()):
        for (key, target_var) in ob.target_vars.iteritems():
            givens[target_var] = ys_shared[key][idx*config.batch_size : (idx+1)*config.batch_size]

    for (key, l_in) in input_layers.iteritems():
        givens[l_in.input_var] = xs_shared[key][idx*config.batch_size:(idx+1)*config.batch_size]

    # sum makes the learning rate independent of batch size!
    train_loss_theano = T.sum(train_losses_theano["objective"]) * (-1 if objectives["train"]["objective"].optimize == MAXIMIZE else 1)

    updates = config.build_updates(train_loss_theano, all_params, learning_rate)

    print "Compiling..."
    iter_train = theano.function([idx],
                                 train_losses_theano.values() + theano_printer.get_the_stuff_to_print(),
                                 givens=givens, on_unused_input="ignore", updates=updates,
                                 # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                                 )

    network_outputs = [
        lasagne.layers.helper.get_output(network_output_layer, deterministic=True)
        for network_output_layer in output_layers.values()
    ]
    iter_predict  = theano.function([idx],
                                    network_outputs + theano_printer.get_the_stuff_to_print(),
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

    last_auc_epoch = -1

    print "Will train for %s epochs" % config.training_data.epochs

    if config.restart_from_save and os.path.isfile(metadata_path):
        print "Load model parameters for resuming"
        resume_metadata = np.load(metadata_path)
        lasagne.layers.set_all_param_values(top_layer, resume_metadata['param_values'])
        start_chunk_idx = resume_metadata['chunks_since_start'] + 1

        # set lr to the correct value
        current_lr = np.float32(utils.current_learning_rate(learning_rate_schedule, start_chunk_idx))
        print "  setting learning rate to %.7f" % current_lr
        learning_rate.set_value(current_lr)
        losses = resume_metadata['losses']
    else:
        start_chunk_idx=0
        losses = dict()
        losses[TRAINING] = dict()
        losses[VALIDATION] = dict()
        for loss_name in train_losses_theano.keys():
            losses[TRAINING][loss_name] = list()

        for dataset_name in config.validation_data.keys():
            losses[VALIDATION][dataset_name] = dict()
            for loss_name in validate_losses_theano.keys():
                losses[VALIDATION][dataset_name][loss_name] = list()


    chunks_train_idcs = itertools.count(start_chunk_idx)
    if config.training_data.epochs:
        num_chunks_train = int(1.0 * config.training_data.epochs * config.training_data.number_of_samples / (config.batch_size * config.batches_per_chunk))
    else:
        num_chunks_train = None


    start_time,prev_time = None,None
    print "Loading first chunks"
    data_load_time = Timer()
    gpu_time = Timer()

    data_load_time.start()
    for e, train_data in izip(chunks_train_idcs, training_data_generator):
        data_load_time.stop()
        if start_time is None:
            start_time = time.time()
            prev_time = start_time

        print
        if num_chunks_train:
            print "Chunk %d/%d" % (e + 1, num_chunks_train)
        else:
            print "Chunk %d" % (e + 1)
        print "=============="
        print "  %s" % config.__name__
        epoch = (1.0 * config.batch_size * config.batches_per_chunk * (e+1) / config.training_data.number_of_samples)
        if epoch>=0.1:
            print "  Epoch %.1f/%s" % (epoch, str(config.training_data.epochs))
        else:
            print "  Epoch %.0e/%s" % (epoch, str(config.training_data.epochs))

        if config.dump_network_loaded_data:
            pickle.dump(train_data, open("data_loader_dump_train_%d.pkl" % e, "wb"))

        if epoch >= last_auc_epoch+1:
            last_auc_epoch = epoch
            print
            for key,ob in objectives["train"].iteritems():
                ob.remove_all_points()

            auc_chunk_generator = config.auc_training_data.generate_batch(
                    chunk_size = chunk_size,
                    required_input = required_input,
                    required_output = required_output,
                )

            print "  AUC-points (%d/%d samples)" % (config.auc_training_data.number_of_used_samples, config.auc_training_data.number_of_samples)
            print "  -----------------------"

            if config.auc_training_data.number_of_samples == 0:
                continue

            chunk_outputs = np.zeros((len(network_outputs),0))
            chunk_labels = np.zeros((0,))
            data_load_time.start()
            for auc_data in buffering.buffered_gen_threaded(auc_chunk_generator):
                data_load_time.stop()
                num_batches_chunk_eval = config.batches_per_chunk

                for key in xs_shared:
                    xs_shared[key].set_value(auc_data["input"][key])

                chunk_labels = np.concatenate((chunk_labels, auc_data["output"]['kaggle-seizure:class']), axis=0)

                for b in xrange(num_batches_chunk_eval):
                    gpu_time.start()
                    th_result = iter_predict(b)
                    gpu_time.stop()
                    resulting_outputs = np.stack(th_result[:len(network_outputs)], axis=0)
                    chunk_outputs = np.concatenate((chunk_outputs, resulting_outputs), axis=1)
                data_load_time.start()
            data_load_time.stop()
            utils.detect_nans(chunk_outputs, xs_shared, ys_shared, all_params)

            for key,ob in objectives["train"].iteritems():
                try:
                    l = config.auc_training_data.number_of_samples
                    ob.add_points(chunk_outputs[0,:l], chunk_labels[:l])
                    print "  Points: %d" % (len(ob.time_added)-2)
                    print "  Current estimated AUC score: %.4f" % ob.current_auc
                    print
                    #ob.print_status()
                except:
                    pass


        for key, rate in learning_rate_schedule.iteritems():
            if epoch >= key:
                lr = np.float32(rate)
                learning_rate.set_value(lr)
        print "  learning rate %.0e" % lr


        for key in xs_shared:
            xs_shared[key].set_value(train_data["input"][key])

        for key in ys_shared:
            if key not in train_data["output"]:
                raise Exception("You forgot to add key %s to OUTPUT_DATA_SIZE_TYPE in your data loader")
            ys_shared[key].set_value(train_data["output"][key])

        chunk_losses = np.zeros((len(train_losses_theano),0))

        for b in xrange(config.batches_per_chunk):
            gpu_time.start()
            th_result = iter_train(b)
            gpu_time.stop()

            resulting_losses = np.stack(th_result[:len(train_losses_theano)], axis=0)

            # these are not needed anyway, just to make Theano call the print function
            # stuff_to_print = th_result[-len(theano_printer.get_the_stuff_to_print()):]
            chunk_losses = np.concatenate((chunk_losses, resulting_losses), axis=1)

        utils.detect_nans(chunk_losses, xs_shared, ys_shared, all_params)

        mean_train_loss = np.mean(chunk_losses, axis=1)
        for loss_name, loss in zip(train_losses_theano.keys(), mean_train_loss):
            losses[TRAINING][loss_name].append(loss)
            print string.rjust(loss_name+":",15), "%.6f" % loss

        validate_every = max(int((config.epochs_per_validation * config.training_data.number_of_samples) / (config.batch_size * config.batches_per_chunk)),1)

        if ((e + 1) % validate_every) == 0 or (num_chunks_train and e+1>=num_chunks_train):
            print
            print "  Validating "
            for dataset_name, dataset_generator in config.validation_data.iteritems():


                validation_chunk_generator = dataset_generator.generate_batch(
                        chunk_size = chunk_size,
                        required_input = required_input,
                        required_output = required_output,
                    )

                print "  %s (%d/%d samples)" % (dataset_name, dataset_generator.number_of_used_samples, dataset_generator.number_of_samples)
                print "  -----------------------"

                if dataset_generator.number_of_samples == 0:
                    continue

                chunk_losses = np.zeros((len(network_outputs),0))
                chunk_labels = np.zeros((0,))

                data_load_time.start()
                for validation_data in buffering.buffered_gen_threaded(validation_chunk_generator):
                    data_load_time.stop()
                    num_batches_chunk_eval = config.batches_per_chunk

                    for key in xs_shared:
                        xs_shared[key].set_value(validation_data["input"][key])

                    chunk_labels = np.concatenate((chunk_labels, validation_data["output"]['kaggle-seizure:class']), axis=0)

                    for b in xrange(num_batches_chunk_eval):
                        gpu_time.start()
                        th_result = iter_predict(b)
                        gpu_time.stop()
                        resulting_losses = np.stack(th_result[:len(network_outputs)], axis=0)
                        chunk_losses = np.concatenate((chunk_losses, resulting_losses), axis=1)
                    data_load_time.start()
                data_load_time.stop()

                utils.detect_nans(chunk_losses, xs_shared, ys_shared, all_params)

                for key,ob in objectives["validate"].iteritems():
                    loss = ob.score_lists(chunk_losses[0,:], chunk_labels)
                    losses[VALIDATION][dataset_name][loss_name].append(loss)
                    print string.rjust(loss_name+":",17), "%.6f" % loss
                print


        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        print "  %s since start (+%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  (%s waiting on gpu vs %s waiting for data)" % (gpu_time, data_load_time)
        try:
            if num_chunks_train:
                est_time_left = time_since_start * (float(num_chunks_train - (e + 1 - start_chunk_idx)) / float(e + 1 - start_chunk_idx))
                eta = datetime.datetime.now() + datetime.timedelta(seconds=est_time_left)
                eta_str = eta.strftime("%c")
                print "  estimated %s to go"  % utils.hms(est_time_left)
                print "  (ETA: %s)" % eta_str
        except OverflowError:
            print "  This will take really long, like REALLY long."

        print "  on average %dms per training sample" % (1000.*time_since_start / ((e+1 - start_chunk_idx) * config.batch_size * config.batches_per_chunk))

        if ((e + 1) % config.save_every_chunks) == 0 or (num_chunks_train and e+1>=num_chunks_train):
            print
            print "Saving metadata, parameters"

            with open(metadata_path, 'w') as f:
                pickle.dump({
                    'metadata_path': metadata_path,
                    'configuration_file': config.__name__,
                    'git_revision_hash': utils.get_git_revision_hash(),
                    'experiment_id': expid,
                    'chunks_since_start': e,
                    'losses': losses,
                    'time_since_start': time_since_start,
                    'param_values': lasagne.layers.get_all_param_values(top_layer)
                }, f, pickle.HIGHEST_PROTOCOL)

            print "  saved to %s" % metadata_path
            print

        gpu_time.reset()
        data_load_time.reset()
        data_load_time.start()

    return





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help='configuration to run',)
    # required = parser.add_argument_group('required arguments')
    # required.add_argument('-c', '--config',
    #                       required=True)
    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(args.config)

    log_file = LOGS_PATH + "%s.log" % expid
    with print_to_file(log_file):

        print "Running configuration:", config.__name__
        print "Current git version:", utils.get_git_revision_hash()

        train_model(expid)
        print "log saved to '%s'" % log_file