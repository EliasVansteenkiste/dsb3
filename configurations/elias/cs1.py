from configurations.default import *

import lasagne
from lasagne.layers import dnn
import numpy as np
from functools import partial

from application.objectives import CrossEntropyObjective, NLLObjective
from application.stage1 import Stage1DataLoader
from application.bcolz_all_data import BcolzAllDataLoader
from application.preprocessors.augment_roi_zero_pad import AugmentROIZeroPad
from interfaces.data_loader import VALIDATION, TRAINING
from application.preprocessors.dicom_to_HU import DicomToHU
from application.preprocessors.normalize_scales import DefaultNormalizer
from interfaces.preprocess import ZMUV
from theano_utils.weight_norm import weight_norm

from deep_learning.nn_lung import Hu2normHULayer, ZMUVLayer

#####################
#   running speed   #
#####################

"This is the number of samples in each batch"
batch_size = 1
"This is the number of batches in each chunk. Computation speeds up if this is as big as possible." \
"However, when too big, the GPU will run out of memory"
batches_per_chunk = 16
"Reload the parameters from last time and continue, or start anew when you run this config file again"
restart_from_save = False
"After how many chunks sho uld you save parameters. Keep this number high for better performance. It will always store at end anyway"
save_every_chunks = 1000. / float(batch_size * batches_per_chunk)

print_gradnorm = False
print_score_every_chunk = True
print_mean_chunks = 800 / (batches_per_chunk*batch_size)
init_weight_norm = 64  # number of samples
dont_sum_losses = True

#####################
#   preprocessing   #
#####################

"Put in here the preprocessors for your data." \
"They will be run consequently on the datadict of the dataloader in the order of your list."

nn_input_shape = (32,)*3
norm_patch_shape = (32,)*3 # in mm
roi_config = "configurations.elias.roi_stage1_1"
max_rois = 64

tag = "bcolzall:"

data_loader = partial(BcolzAllDataLoader, #BcolzAllDataLoader, Stage1DataLoader
    multiprocess=True,
    crash_on_exception=True)

augment_roi = partial(AugmentROIZeroPad,
    max_rois=max_rois,
    roi_config=roi_config,
    tags=[tag+"3d"],
    output_shape=nn_input_shape,
    norm_patch_shape=norm_patch_shape,
    interp_order=1,
    mode="constant"
    )

preprocessors = [
    # DicomToHU(tags=[tag+"3d"]),
    augment_roi(
        augmentation_params={
            "scale": [1, 1, 1],  # factor
            "uniform scale": 1,  # factor
            "rotation": [180, 180, 180],  # degrees
            "shear": [0, 0, 0],  # degrees
            "translation": [5, 5, 5],  # mm
            "reflection": [0.5, 0.5, 0.5]},  # Bernoulli p
        ),
    # DefaultNormalizer(tags=[tag+"3d"])
    # ZMUV(tag+"3d", bias=-648.59027, std=679.21021)
]

preprocessors_valid = [
    # DicomToHU(tags=[tag+"3d"]),
    augment_roi(),
    # DefaultNormalizer(tags=[tag+"3d"])
    # ZMUV(tag+"3d", bias=-648.59027, std=679.21021)
]

#####################
#     training      #
#####################

"This is the train dataloader. We will train until this one stops loading data."
"You can set the number of epochs, the datasets and if you want it multiprocessed"
n_epochs = 100
training_data = data_loader(
    sets=TRAINING,
    epochs=n_epochs,
    preprocessors=preprocessors)

"Schedule the reducing of the learning rate. On indexing with the number of epochs, it should return a value for the learning rate."
# lr = 0.0001
# lr_min = lr / 1000.
# lr_decay = 0.95
# learning_rate_schedule = {}
# for i in range(n_epochs):
#     lr_ = lr * (lr_decay ** i)
#     if lr_ < lr_min: break
#     learning_rate_schedule[i] = lr_

learning_rate_schedule = {
    0: 1e-4,
    int(n_epochs * 0.5): 5e-5,
    int(n_epochs * 0.6): 2.5e-5,
    int(n_epochs * 0.7): 1.25e-5,
    int(n_epochs * 0.8): 0.625e-6,
    int(n_epochs * 0.9): 0.3125e-6
}

# print learning_rate_schedule

"The function to build updates."
build_updates = lasagne.updates.adam

#####################
#    validation     #
#####################
"We do our validation after every x epochs of training"
epochs_per_validation = 2.0

"Which data do we want to validate on. We will run all validation objectives on each validation data set."
validation_data = {
    "validation set": data_loader(sets=VALIDATION,
                                 epochs=1,
                                 preprocessors=preprocessors_valid,
                                 process_last_chunk=True),
    "training set": data_loader(sets=TRAINING,
                               epochs=0.01,
                               preprocessors=preprocessors_valid,
                               process_last_chunk=True)
}

#####################
#      testing      #
#####################
"This is the data which will be used for testing."
test_data = None

#####################
#     objective     #
#####################

"Here we return a dict with the Theano objectives we are interested in. Both for the train and validation set."
"On both sets, you may request multiple objectives! Only the one called 'objective' is used to optimize on."


def build_objectives(interface_layers):
    obj = NLLObjective(interface_layers["outputs"], target_name=tag[:-1])
    # obj = CrossEntropyObjective(interface_layers["outputs"], target_name=tag[:-1])
    return {
        "train": {
            "objective": obj,
        },
        "validate": {
            "objective": obj,
        }
    }


#################
# Regular model #
#################

"For ease of working, we predefine some layers with their parameters"
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=lasagne.init.Orthogonal('relu'),
                 b=lasagne.init.Constant(0.0),
                 # untie_biases=True,
                 nonlinearity=lasagne.nonlinearities.rectify)

conv2d = partial(dnn.Conv2DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=lasagne.init.Orthogonal('relu'),
                 b=lasagne.init.Constant(0.0),
                 # untie_biases=True,
                 nonlinearity=lasagne.nonlinearities.rectify)



max_pool2d = partial(dnn.MaxPool2DDNNLayer, pool_size=2)

max_pool3d = partial(dnn.MaxPool3DDNNLayer, pool_size=2)

dense = partial(lasagne.layers.DenseLayer,
                # W=lasagne.init.Orthogonal('relu'),
                b=lasagne.init.Constant(0.0),
                nonlinearity=lasagne.nonlinearities.rectify)

nin = partial(lasagne.layers.NINLayer,
              W=lasagne.init.Orthogonal('relu'),
              b=lasagne.init.Constant(0.0),
              nonlinearity=lasagne.nonlinearities.rectify)

drop = lasagne.layers.DropoutLayer

# bn = lasagne.layers.batch_norm
bn = lasagne.layers.dnn.batch_norm_dnn
wn = weight_norm


def inrn_v2(lin):
    n_base_filter = 32

    l1 = conv3d(lin, n_base_filter, filter_size=1)

    l2 = conv3d(lin, n_base_filter, filter_size=1)
    l2 = conv3d(l2, n_base_filter, filter_size=3)

    l3 = conv3d(lin, n_base_filter, filter_size=1)
    l3 = conv3d(l3, n_base_filter, filter_size=3)
    l3 = conv3d(l3, n_base_filter, filter_size=3)

    l = lasagne.layers.ConcatLayer([l1, l2, l3])

    l = conv3d(l, lin.output_shape[1], filter_size=1)

    l = lasagne.layers.ElemwiseSumLayer([l, lin])

    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)

    return l

def inrn_v2_red(lin):
    # We want to reduce our total volume /4

    den = 16
    nom2 = 4
    nom3 = 5
    nom4 = 7

    ins = lin.output_shape[1]

    l1 = max_pool3d(lin)

    l2 = conv3d(lin, ins // den * nom2, filter_size=3, stride=2)

    l3 = conv3d(lin, ins // den * nom2, filter_size=1)
    l3 = conv3d(l3, ins // den * nom3, filter_size=3, stride=2)

    l4 = conv3d(lin, ins // den * nom2, filter_size=1)
    l4 = conv3d(l4, ins // den * nom3, filter_size=3)
    l4 = conv3d(l4, ins // den * nom4, filter_size=3, stride=2)

    l = lasagne.layers.ConcatLayer([l1, l2, l3, l4])

    return l



"Here we build a model. The model returns a dict with the requested inputs for each layer:" \
"And with the outputs it generates. You may generate multiple outputs (for analysis or for some other objectives, etc)" \
"Unused outputs don't cost in performance"


def build_model():
    l_in = lasagne.layers.InputLayer(shape=(None, max_rois) + nn_input_shape)
    l_norm = Hu2normHULayer(l_in, min_hu=-1000, max_hu=400)

    l = lasagne.layers.ReshapeLayer(l_norm, (-1, 1) + nn_input_shape)

    l = conv3d(l, 64)
    l = inrn_v2_red(l)
    l = inrn_v2_red(l)
    l = inrn_v2_red(l)

    l = dense(drop(l), 64)
    l = lasagne.layers.DenseLayer(l, num_units=2,
                                 W=lasagne.init.Constant(0.),
                                 nonlinearity=lasagne.nonlinearities.softmax)

    l = lasagne.layers.ReshapeLayer(l, (-1, max_rois,2))

    l = dense(drop(l), 64)

    l_out = lasagne.layers.DenseLayer(l, num_units=2,
                                 W=lasagne.init.Constant(0.),
                                 nonlinearity=lasagne.nonlinearities.softmax)

    return {
        "inputs": {
            tag+"3d": l_in,
        },
        "outputs": {
            "predicted_probability": l_out
        },
    }
