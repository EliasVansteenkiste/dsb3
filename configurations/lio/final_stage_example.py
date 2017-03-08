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
            "rotation": [0, 0, 180],  # degrees
            "shear": [0, 0, 0],  # degrees
            "translation": [5, 5, 5],  # mm
            "reflection": [0, 0, 0]},  # Bernoulli p
        ),
    # DefaultNormalizer(tags=[tag+"3d"])
    ZMUV(tag+"3d", bias=-648.59027, std=679.21021)
]

preprocessors_valid = [
    # DicomToHU(tags=[tag+"3d"]),
    augment_roi(),
    # DefaultNormalizer(tags=[tag+"3d"])
    ZMUV(tag+"3d", bias=-648.59027, std=679.21021)
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
lr = 0.00001
lr_min = lr / 1000.
lr_decay = 0.95
learning_rate_schedule = {}
for i in range(n_epochs):
    lr_ = lr * (lr_decay ** i)
    if lr_ < lr_min: break
    learning_rate_schedule[i] = lr_

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

"Here we build a model. The model returns a dict with the requested inputs for each layer:" \
"And with the outputs it generates. You may generate multiple outputs (for analysis or for some other objectives, etc)" \
"Unused outputs don't cost in performance"


def build_model():
    l_in = lasagne.layers.InputLayer(shape=(None, max_rois) + nn_input_shape)

    l = lasagne.layers.DimshuffleLayer(l_in, pattern=(0, 1, 4, "x", 3, 2))

    l = lasagne.layers.ReshapeLayer(l, (-1, 1, nn_input_shape[0], nn_input_shape[1]))

    n = 64
    l = conv2d(l, n, filter_size=5, stride=2)
    l = wn(l)

    n *= 2
    l = conv2d(l, n)
    l = wn(l)
    l = conv2d(l, n)
    l = wn(l)
    l = max_pool2d(l)

    n *= 2
    l = conv2d(l, n)
    l = wn(l)
    l = conv2d(l, n)
    l = wn(l)
    l = max_pool2d(l)

    n_features = np.prod(l.output_shape[-3:])
    l = lasagne.layers.ReshapeLayer(l, (-1, max_rois*nn_input_shape[2], n_features))
    l = lasagne.layers.FeaturePoolLayer(l, max_rois*nn_input_shape[2], axis=1)

    n *= 2
    l = dense(l, n)
    l = wn(l)
    l = drop(l)
    l = dense(l, n)
    l = wn(l)
    l = drop(l)

    # l = lasagne.layers.DenseLayer(l,
    #                               num_units=1,
    #                               W=lasagne.init.Constant(0.),
    #                               b=lasagne.init.Constant(-np.log(1. / 0.25 - 1.)),
    #                               nonlinearity=lasagne.nonlinearities.sigmoid)
    # l_out = lasagne.layers.reshape(l, shape=(-1,))

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
