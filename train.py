"""
This is the script which trains your model on the dataset according to a configuration file
The resulting parameters are consequently stored, and can be used by other scripts.

Run with:
python train.py myconfigfile
"""
import argparse
from collections import defaultdict
from functools import partial
from itertools import izip
import cPickle as pickle
import string
import datetime
import itertools
import lasagne
import time
from interfaces.data_loader import VALIDATION, VALID_SAMPLES
from interfaces.data_loader import TRAINING
from interfaces.objectives import MAXIMIZE
from utils.log import print_to_file

from utils.configuration import set_configuration, config, get_configuration_name
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
# this removes some annoying messages from being printed
warnings.simplefilter("error")
import sys
# because python does not like recursion, but we do
sys.setrecursionlimit(10000)


def train_model(expid):
    """
    This function trains the model, and will use the name expid to store and report the results
    :param expid: the name
    :return:
    """
    metadata_path = MODEL_PATH + "%s.pkl" % expid

    # Fast_run is very slow, but might be better of debugging.
    # Make sure you don't leave it on accidentally!
    if theano.config.optimizer != "fast_run":
        print "WARNING: not running in fast mode!"

    print "Build model"
    # Get the input and output layers of our model
    interface_layers = config.build_model()

    output_layers = interface_layers["outputs"]
    input_layers = interface_layers["inputs"]

    # merge all output layers into a fictional dummy layer which is not actually used
    top_layer = lasagne.layers.MergeLayer(
        incomings=output_layers.values()
    )
    # get all the trainable parameters from the model
    all_layers = lasagne.layers.get_all_layers(top_layer)
    all_params = lasagne.layers.get_all_params(top_layer, trainable=True)

    # do not train beyond the layers in cutoff_gradients. Remove all their parameters from the optimization process
    if "cutoff_gradients" in interface_layers:
        submodel_params = [param for value in interface_layers["cutoff_gradients"] for param in lasagne.layers.get_all_params(value)]
        all_params = [p for p in all_params if p not in submodel_params]

    # some parameters might already be pretrained! Load their values from the requested configuration name.
    if "pretrained" in interface_layers:
        for config_name, layers_dict in interface_layers["pretrained"].iteritems():
            pretrained_metadata_path = MODEL_PATH + "%s.pkl" % config_name
            pretrained_resume_metadata = np.load(pretrained_metadata_path)
            pretrained_top_layer = lasagne.layers.MergeLayer(
                incomings = layers_dict.values()
            )
            lasagne.layers.set_all_param_values(pretrained_top_layer, pretrained_resume_metadata['param_values'])

    # Count all the parameters we are actually optimizing, and visualize what the model looks like.

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

    num_params = sum([np.prod(p.get_value().shape) for p in all_params])
    print "  number of parameters:", comma_seperator(num_params)

    # Build all the objectives requested by the configuration
    objectives = config.build_objectives(interface_layers)

    train_losses_theano    = {key:ob.get_loss()
                                for key,ob in objectives["train"].iteritems()}

    validate_losses_theano = {key:ob.get_loss(deterministic=True)
                                for key,ob in objectives["validate"].iteritems()}

    # Create the Theano variables necessary to interface with the models
    # the input:
    xs_shared = {
        key: lasagne.utils.shared_empty(dim=len(l_in.output_shape), dtype='float32')
        for (key, l_in) in input_layers.iteritems()
    }

    # the output:
    ys_shared = {
        key: lasagne.utils.shared_empty(dim=target_var.ndim, dtype=target_var.dtype)
        for (_,ob) in itertools.chain(objectives["train"].iteritems(), objectives["validate"].iteritems())
        for (key, target_var) in ob.target_vars.iteritems()
    }

    # Set up the learning rate schedule
    learning_rate_schedule = config.learning_rate_schedule
    learning_rate = theano.shared(np.float32(learning_rate_schedule[0]))


    # We only work on one batch at the time on our chunk. Set up the Theano code which does this
    idx = T.lscalar('idx')  # the value representing the number of the batch we are currently into our chunk of data

    givens = dict()
    for (_,ob) in itertools.chain(objectives["train"].iteritems(), objectives["validate"].iteritems()):
        for (key, target_var) in ob.target_vars.iteritems():
            givens[target_var] = ys_shared[key][idx*config.batch_size : (idx+1)*config.batch_size]

    for (key, l_in) in input_layers.iteritems():
        givens[l_in.input_var] = xs_shared[key][idx*config.batch_size:(idx+1)*config.batch_size]

    # sum over the losses of the objective we optimize. We will optimize this sum (either minimize or maximize)
    # sum makes the learning rate independent of batch size!
    if hasattr(config, "dont_sum_losses") and config.dont_sum_losses:
        train_loss_theano = T.mean(train_losses_theano["objective"])
    else:
        train_loss_theano = T.sum(train_losses_theano["objective"]) * (-1 if objectives["train"]["objective"].optimize == MAXIMIZE else 1)

    # build the update step for Theano
    updates = config.build_updates(train_loss_theano, all_params, learning_rate)

    if hasattr(config, "print_gradnorm") and config.print_gradnorm:
        all_grads = theano.grad(train_loss_theano, all_params, disconnected_inputs='warn')
        grad_norm = T.sqrt(T.sum([(g ** 2).sum() for g in all_grads]) + 1e-9)
        grad_norm.name = "grad_norm"
        theano_printer.print_me_this("  grad norm", grad_norm)
        # train_losses_theano["grad_norm"] = grad_norm


    # Compile the Theano function of your model+objective
    print "Compiling..."
    iter_train = theano.function([idx],
                                 train_losses_theano.values() + theano_printer.get_the_stuff_to_print(),
                                 givens=givens, on_unused_input="ignore", updates=updates,
                                 # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                                 )

    if hasattr(config, "print_gradnorm") and config.print_gradnorm:
        del theano_printer._stuff_to_print[-1]

    # For validation, we also like to have something which returns the output of our model without the objective
    network_outputs = [
        lasagne.layers.helper.get_output(network_output_layer, deterministic=True)
        for network_output_layer in output_layers.values()
    ]
    iter_predict  = theano.function([idx],
                                    network_outputs + theano_printer.get_the_stuff_to_print(),
                                    givens=givens, on_unused_input="ignore")

    # The data loader will need to know which kinds of data it actually needs to load
    # collect all the necessary tags for the model.
    required_input = {
        key: l_in.output_shape
        for (key, l_in) in input_layers.iteritems()
    }
    required_output = {
        key: None  # size is not needed
        for (_,ob) in itertools.chain(objectives["train"].iteritems(), objectives["validate"].iteritems())
        for (key, target_var) in ob.target_vars.iteritems()
    }

    # The data loaders need to prepare before they should start
    # This is usually where the data is loaded from disk onto memory
    print "Preparing dataloaders"
    config.training_data.prepare()
    for validation_data in config.validation_data.values():
        validation_data.prepare()

    print "Will train for %s epochs" % config.training_data.epochs

    # If this is the second time we run this configuration, we might need to load the results of the previous
    # optimization. Check if this is the case, and load the parameters and stuff. If not, start from zero.
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
        config.training_data.skip_first_chunks(start_chunk_idx)
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


    # Make a data generator which returns preprocessed chunks of data which are fed to the model
    # Note that this is a generator object! It is a special kind of iterator.
    chunk_size = config.batches_per_chunk * config.batch_size

    # Weight normalization
    if hasattr(config, "init_weight_norm") and not config.restart_from_save:
        theano_printer._stuff_to_print = []
        from theano_utils.weight_norm import train_weight_norm
        train_weight_norm(config, output_layers, all_layers, idx, givens, xs_shared, chunk_size, required_input, required_output)


    training_data_generator = buffering.buffered_gen_threaded(
        config.training_data.generate_batch(
            chunk_size = chunk_size,
            required_input = required_input,
            required_output = required_output,
        )
    )

    # Estimate the number of batches we will train for.
    chunks_train_idcs = itertools.count(start_chunk_idx)
    if config.training_data.epochs:
        num_chunks_train = int(1.0 * config.training_data.epochs * config.training_data.number_of_samples / (config.batch_size * config.batches_per_chunk))
    else:
        num_chunks_train = None

    # Start the timer objects
    start_time,prev_time = None,None
    print "Loading first chunks"
    data_load_time = Timer()
    gpu_time = Timer()

    #========================#
    # This is the train loop #
    #========================#
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

        # Estimate the current epoch we are at
        epoch = (1.0 * config.batch_size * config.batches_per_chunk * (e+1) / config.training_data.number_of_samples)
        if epoch>=0.1:
            print "  Epoch %.1f/%s" % (epoch, str(config.training_data.epochs))
        else:
            print "  Epoch %.0e/%s" % (epoch, str(config.training_data.epochs))

        # for debugging the data loader, it might be useful to dump everything it loaded and analyze it.
        if config.dump_network_loaded_data:
            pickle.dump(train_data, open("data_loader_dump_train_%d.pkl" % e, "wb"))

        # Update the learning rate with the new epoch the number
        for key, rate in learning_rate_schedule.iteritems():
            if epoch >= key:
                lr = np.float32(rate)
                learning_rate.set_value(lr)
        print "  learning rate %.0e" % lr

        # Move this data from the data loader onto the Theano variables
        for key in xs_shared:
            xs_shared[key].set_value(train_data["input"][key])

        for key in ys_shared:
            if key not in train_data["output"]:
                raise Exception("You forgot to add key %s to OUTPUT_DATA_SIZE_TYPE in your data loader"%key)
            ys_shared[key].set_value(train_data["output"][key])

        # loop over all the batches in one chunk, and keep the losses
        chunk_losses = np.zeros((len(train_losses_theano),0))
        for b in xrange(config.batches_per_chunk):
            gpu_time.start()
            th_result = iter_train(b)
            gpu_time.stop()

            resulting_losses = np.stack(th_result[:len(train_losses_theano)], axis=0)

            # these are not needed anyway, just to make Theano call the print function
            # stuff_to_print = th_result[-len(theano_printer.get_the_stuff_to_print()):]
            # print resulting_losses.shape, chunk_losses.shape
            chunk_losses = np.concatenate((chunk_losses, resulting_losses), axis=1)

        # check if we found NaN's. When there are NaN's we might as well exit.
        utils.detect_nans(chunk_losses, xs_shared, ys_shared, all_params)

        # Average our losses, and print them.
        mean_train_loss = np.mean(chunk_losses, axis=1)
        for loss_name, loss in zip(train_losses_theano.keys(), mean_train_loss):
            losses[TRAINING][loss_name].append(loss)
            print string.rjust(loss_name+":",15), "%.6f" % loss

        # Now, we will do validation. We do this about every config.epochs_per_validation epochs.
        # We also always validate at the end of every training!
        validate_every = max(int((config.epochs_per_validation * config.training_data.number_of_samples) / (config.batch_size * config.batches_per_chunk)),1)

        if ((e+1) % validate_every) == 0 or (num_chunks_train and e+1>=num_chunks_train):
            print
            print "  Validating "

            # We might test on multiple datasets, such as the Train set, Validation set, ...
            for dataset_name, dataset_generator in config.validation_data.iteritems():

                # Start loading the validation data!
                validation_chunk_generator = dataset_generator.generate_batch(
                        chunk_size = chunk_size,
                        required_input = required_input,
                        required_output = required_output,
                    )

                print "  %s (%d/%d samples)" % (dataset_name,
                                                dataset_generator.number_of_samples_in_iterator,
                                                dataset_generator.number_of_samples)
                print "  -----------------------"

                # If there are no validation samples, don't bother validating.
                if dataset_generator.number_of_samples == 0:
                    continue

                validation_predictions = None

                # Keep the labels of the validation data for later.
                output_keys_to_store = set()
                losses_to_store = dict()
                for key,ob in objectives["validate"].iteritems():
                    if ob.mean_over_samples:
                        losses_to_store[key] = None
                    else:
                        output_keys_to_store.add(ob.target_key)
                chunk_labels = {k:None for k in output_keys_to_store}
                store_network_output = (len(output_keys_to_store)>0)

                # loop over all validation data chunks
                data_load_time.start()
                for validation_data in buffering.buffered_gen_threaded(validation_chunk_generator):
                    data_load_time.stop()
                    num_batches_chunk_eval = config.batches_per_chunk

                    # set the validation data to the required Theano variables. Note, there is no
                    # use setting the output variables, as we do not have labels of the validation set!
                    for key in xs_shared:
                        xs_shared[key].set_value(validation_data["input"][key])

                    # store all the output keys required for finding the validation error
                    for key in output_keys_to_store:
                        new_data = validation_data["output"][key][:validation_data["valid_samples"]]

                        if chunk_labels[key] is None:
                            chunk_labels[key] = new_data
                        else:
                            chunk_labels[key] = np.concatenate((chunk_labels[key], new_data), axis=0)

                    # loop over the batches of one chunk, and keep the predictions
                    chunk_predictions = None
                    for b in xrange(num_batches_chunk_eval):
                        gpu_time.start()
                        th_result = iter_predict(b)
                        gpu_time.stop()
                        resulting_predictions = np.stack(th_result[:len(network_outputs)], axis=0)
                        assert len(network_outputs)==1, "Multiple outputs not implemented yet"
                        if chunk_predictions is None:
                            chunk_predictions = resulting_predictions
                        else:
                            chunk_predictions = np.concatenate((chunk_predictions, resulting_predictions), axis=1)


                    # Check for NaN's. Panic if there are NaN's during validation.
                    utils.detect_nans(chunk_predictions, xs_shared, ys_shared, all_params)

                    # add the predictions of this chunk, to the global predictions (if needed)
                    if chunk_predictions is not None:
                        chunk_predictions = chunk_predictions[:validation_data[VALID_SAMPLES]]
                        if store_network_output:
                            if validation_predictions is None:
                                validation_predictions = chunk_predictions
                            else:
                                validation_predictions = np.concatenate((validation_predictions, chunk_predictions), axis=1)

                    # if you can calculate the losses per chunk, and take the mean afterwards, do that.
                    for key,ob in objectives["validate"].iteritems():
                        if ob.mean_over_samples:
                            new_losses = []
                            for i in xrange(validation_data[VALID_SAMPLES]):
                                loss = ob.get_loss_from_lists(
                                    chunk_predictions[0,i:i+1],
                                    validation_data["output"][ob.target_key][i:i+1]
                                )
                                new_losses.append(loss)

                            new_losses = np.array(new_losses)
                            if losses_to_store[key] is None:
                                losses_to_store[key] = new_losses
                            else:
                                losses_to_store[key] = np.concatenate((losses_to_store[key], new_losses), axis=0)


                    data_load_time.start()
                data_load_time.stop()

                # Compare the predictions with the actual labels and print them.
                for key,ob in objectives["validate"].iteritems():
                    if ob.mean_over_samples:
                        loss = np.mean(losses_to_store[key])
                    else:
                        loss = ob.get_loss_from_lists(validation_predictions[0,:], chunk_labels[ob.target_key])
                    losses[VALIDATION][dataset_name][key].append(loss)
                    print string.rjust(key+":",17), "%.6f" % loss
                print

        # Good, we did one chunk. Let us check how much time this took us. Print out some stats.
        now = time.time()
        time_since_start = now - start_time
        time_since_prev = now - prev_time
        prev_time = now
        # This is the most useful stat of all! Keep this number low, and your total optimization time will be low too.
        print "  on average %dms per training sample" % (1000.*time_since_start / ((e+1 - start_chunk_idx) * config.batch_size * config.batches_per_chunk))
        print "  %s since start (+%.2f s)" % (utils.hms(time_since_start), time_since_prev)
        print "  %s waiting on gpu vs %s waiting for data" % (gpu_time, data_load_time)
        try:
            if num_chunks_train:  # only if we ever stop running
                est_time_left = time_since_start * (float(num_chunks_train - (e + 1 - start_chunk_idx)) / float(e + 1 - start_chunk_idx))
                eta = datetime.datetime.now() + datetime.timedelta(seconds=est_time_left)
                eta_str = eta.strftime("%c")
                print "  estimated %s to go"  % utils.hms(est_time_left)
                print "  (ETA: %s)" % eta_str
                if hasattr(config, "print_mean_chunks"):
                    avg_train = losses[TRAINING]["objective"]
                    n = min(len(avg_train), config.print_mean_chunks)
                    avg_train = avg_train[-n:]
                    print "  mean loss last %i chunks: %.3f"%(n, np.mean(avg_train))
        except OverflowError:
            # Shit happens
            print "  This will take really long, like REALLY long."
        if hasattr(config, "print_score_every_chunk") and config.print_score_every_chunk\
                and len(losses[VALIDATION]["training set"]["objective"]) > 0:
            print "  train: best %.3f latest %.3f, valid: best %.3f latest %.3f " % (
                np.min(losses[VALIDATION]["training set"]["objective"]),
                losses[VALIDATION]["training set"]["objective"][-1],
                np.min(losses[VALIDATION]["validation set"]["objective"]),
                losses[VALIDATION]["validation set"]["objective"][-1]
            )

        # Save the data every config.save_every_chunks chunks. Or at the end of the training.
        # We should make it config.save_every_epochs epochs sometimes. Consistency
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

        # reset the timers for next round. This needs to happen here, because at the end of the big for loop
        # we already want te get a chunk immediately for the next loop. The iterator is an argument of the for loop.
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

    expid = utils.generate_expid(get_configuration_name())

    log_file = LOGS_PATH + "%s-train.log" % expid
    with print_to_file(log_file):

        print "Running configuration:", config.__name__
        print "Current git version:", utils.get_git_revision_hash()

        train_model(expid)
        print "log saved to '%s'" % log_file
        sys.stdout.flush()