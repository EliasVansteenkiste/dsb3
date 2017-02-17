"""
Usage: python scripts/lio/roi.py <ROI config>


Example:
python scripts/lio/roi.py configurations/lio/roi_stage1.py

-> ROIs are saved in paths.MODEL_PREDICTIONS_PATH
"""

import matplotlib
matplotlib.use('Agg')

import utils
import utils.plt
from utils import LOGS_PATH, MODEL_PATH, MODEL_PREDICTIONS_PATH, paths
from utils.log import print_to_file
from utils.configuration import set_configuration, config, get_configuration_name
from interfaces.data_loader import TRAIN, VALIDATION, TEST, INPUT
from utils.transformation_3d import affine_transform, apply_affine_transform





import argparse
import theano
import numpy as np
import theano.tensor as T
import os
import sys
import cPickle as pickle
import string
import lasagne
import time
from itertools import product

sys.path.append(".")
from theano_utils import theano_printer

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

    print "Preparing dataloaders"
    config.data_loader.prepare()

    print "Load model parameters"
    metadata = np.load(metadata_path)
    lasagne.layers.set_all_param_values(top_layer, metadata['param_values'])

    start_time, prev_time = None, None

    import multiprocessing as mp
    jobs = []

    idx=0
    num_candidates=[]
    for set in [VALIDATION, TRAIN, TEST]:
        set_indices = config.data_loader.indices[set]
        num_candidates[idx]=0
        for _i, sample_id in enumerate(set_indices):

            if start_time is None:
                start_time = time.time()
                prev_time = start_time
            print "sample_id", sample_id, _i+1, "/", len(set_indices), "in", set

            filenametag = input_layers.keys()[0].split(":")[0] + ":patient_id"
            data = config.data_loader.load_sample(sample_id,
                                                  input_layers.keys()+config.extra_tags+[filenametag],{})

            patient_id = data["input"][filenametag]
            print patient_id
            seg_shape = output_layers["predicted_segmentation"].output_shape[1:]
            patch_gen = patch_generator(data, seg_shape, input_layers.keys()[0].split(":")[0]+":")
            t0 = time.time()
            preds = []
            patches = []
            for patch_idx, patch in enumerate(patch_gen):
                for key in xs_shared:
                    xs_shared[key].set_value(patch[key][None,:])

                print " patch_generator", time.time() - t0

                t0 = time.time()
                th_result = iter_test(0)
                print " iter_test", time.time()-t0

                predictions = th_result[:len(network_outputs)]

                preds.append(predictions[0][0])
                patches.append(patch[xs_shared.keys()[0]])
                t0 = time.time()

            pred = glue_patches(preds)

            if not config.plot and config.multiprocess:
                jobs = [job for job in jobs if job.is_alive]
                if len(jobs) >= 3:
                    # print "waiting", len(jobs)
                    jobs[0].join()
                    del jobs[0]
                jobs.append(mp.Process(target=extract_nodules, args=((pred, patient_id, expid),) ) )
                jobs[-1].daemon=True
                jobs[-1].start()
            else:
                rois = extract_nodules((pred, patient_id, expid))
                print "patient", patient_id, len(rois), "nodules"
                num_candidates[idx]+=len(rois)

            print "candidates per patient: {}".format(len(rois))

            now = time.time()
            time_since_start = now - start_time
            time_since_prev = now - prev_time
            prev_time = now
            print "  %s since start (+%.2f s)" % (utils.hms(time_since_start), time_since_prev)

            if config.plot:
                plot_segmentation_and_nodules(patches, rois, pred, patient_id)
        idx=idx+1

    print "number candidates: {}".format(num_candidates)

    return


def plot_segmentation_and_nodules(patches, rois, pred, patient_id):
    dir_path = paths.ANALYSIS_PATH + expid
    if not os.path.exists(dir_path): os.mkdir(dir_path)
    for i, patch in enumerate(patches):
        patches[i] = patch[10:-10, 10:-10, 10:-10]
    dat = glue_patches(patches)
    roi_vol = np.zeros_like(dat)
    if rois is not None:
        x, y, z = np.ogrid[:roi_vol.shape[0], :roi_vol.shape[1], :roi_vol.shape[2]]
        for roi in rois:
            distance2 = ((x - roi[0]) ** 2 + (y - roi[1]) ** 2 + (z - roi[2]) ** 2)
            roi_vol[(distance2 <= 5)] = 1

    utils.plt.cross_sections([dat, pred, roi_vol], save=dir_path + "/roi%s.jpg" % patient_id, normalize=False)


def extract_nodules((pred, patient_id, expid)):
    t0 = time.time()

    rois = config.extract_nodules(pred)

    if rois is None:
        print " extract_nodules", 0, time.time() - t0
    else:
        print " extract_nodules", len(rois), time.time() - t0
        dir_path = MODEL_PREDICTIONS_PATH + expid
        if not os.path.exists(dir_path): os.mkdir(dir_path)
        with open(dir_path+"/%s.pkl"%patient_id, 'w') as f:
            pickle.dump(rois, f, pickle.HIGHEST_PROTOCOL)

    return rois


stride = None
patch_count = None
norm_shape = None

def patch_generator(sample, segmentation_shape, tag):
    global patch_count, stride, norm_shape

    for prep in config.preprocessors: prep.process(sample)

    data = sample[INPUT][tag + "3d"]
    spacing = sample[INPUT][tag + "pixelspacing"]
    labels=sample[INPUT][tag + "labels"]
    print "XXXXX Printing out labels XXXXX:"
    print labels

    input_shape = np.asarray(data.shape, np.float)
    pixel_spacing = np.asarray(spacing, np.float)
    output_shape = np.asarray(config.patch_shape, np.float)
    mm_patch_shape = np.asarray(config.norm_patch_shape, np.float)
    stride = np.asarray(segmentation_shape, np.float) * mm_patch_shape / output_shape

    norm_shape = input_shape * pixel_spacing
    _patch_shape = norm_shape * output_shape / mm_patch_shape

    patch_count = np.ceil(norm_shape / stride).astype("int")
    print "patch_count", patch_count
    print "stride", stride
    print spacing
    print norm_shape

    for x,y,z in product(range(patch_count[0]), range(patch_count[1]), range(patch_count[2])):

        offset = np.array([stride[0]*x, stride[1]*y, stride[2]*z], np.float)
        print (x*patch_count[1]*patch_count[2] + y*patch_count[2] +z), "/", np.prod(patch_count), (x,y,z)

        shift_center = affine_transform(translation=-(input_shape / 2. - 0.5))
        normscale = affine_transform(scale=norm_shape / input_shape)
        offset_patch = affine_transform(translation=norm_shape/2. - 0.5 - offset-(stride/2.0-0.5))# - (mm_patch_shape - segmentation_shape)*norm_shape/_patch_shape -segmentation_shape*norm_shape/_patch_shape/2.)
        patchscale = affine_transform(scale=_patch_shape / norm_shape)
        unshift_center = affine_transform(translation=output_shape / 2. - 0.5)
        matrix = shift_center.dot(normscale).dot(offset_patch).dot(patchscale).dot(unshift_center)
        output = apply_affine_transform(data, matrix, output_shape=output_shape.astype(np.int))


        patch = {}
        patch[tag+"3d"] = output
        patch["offset"] = offset
        s = {INPUT: patch}
        for prep in config.postpreprocessors: prep.process(s)
        yield patch


def glue_patches(p):
    global patch_count, stride, norm_shape

    preds = []
    for x in range(patch_count[0]):
        preds_y = []
        for y in range(patch_count[1]):
            ofs = y * patch_count[2] + x * patch_count[2] * patch_count[1]
            preds_z = np.concatenate(p[ofs:ofs + patch_count[2]], axis=2)
            preds_y.append(preds_z)
        preds_y = np.concatenate(preds_y, axis=1)
        preds.append(preds_y)

    preds = np.concatenate(preds, axis=0)
    preds = preds[:int(round(norm_shape[0])), :int(round(norm_shape[1])), :int(round(norm_shape[2]))]
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help='configuration to run',)
    args = parser.parse_args()
    set_configuration(args.config)

    expid = utils.generate_expid(get_configuration_name())

    log_file = LOGS_PATH + "%s-train.log" % expid
    with print_to_file(log_file):

        print "Running configuration:", config.__name__
        print "Current git version:", utils.get_git_revision_hash()

        extract_rois(expid)
        print "log saved to '%s'" % log_file
