from configurations.default import *

import lasagne
from lasagne.layers import dnn
import theano.tensor as T
import numpy as np
from functools import partial

from application.objectives import CrossEntropyObjective
from application.bcolz_all_data import BcolzAllDataLoader
# from application.stage1 import Stage1DataLoader
from interfaces.data_loader import VALIDATION, TRAINING, TEST, TRAIN
from application.preprocessors.augmentation_3d import Augment3D
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
save_every_chunks = 1000. / float(batch_size*batches_per_chunk)

multiprocessing_on = True
print_gradnorm = False
print_score_every_chunk = True
print_mean_chunks = 800 / batches_per_chunk
init_weight_norm = 32 # number of samples


#####################
#   preprocessing   #
#####################

"Put in here the preprocessors for your data." \
"They will be run consequently on the datadict of the dataloader in the order of your list."
nn_input_shape = (128, 128, 40)
norm_patch_shape = (300, 300, 280)

preprocessors = [
    # DicomToHU(tags=["bcolzall:3d"]),
    Augment3D(
        tags=["bcolzall:3d"],
        output_shape=nn_input_shape,
        norm_patch_shape=norm_patch_shape,
        augmentation_params={
            "scale": [1, 1, 1],  # factor
            "uniform scale": 1,  # factor
            "rotation": [0, 0, 180],  # degrees
            "shear": [0, 0, 0],  # degrees
            "translation": [50, 50, 50],  # mm
            "reflection": [.5, .5, .5]},  # Bernoulli p
        interp_order=1,
        mode="constant"),
    # DefaultNormalizer(tags=["bcolzall:3d"])
    ZMUV("bcolzall:3d", bias=-648.59027, std=679.21021)
]

preprocessors_valid = [
    # DicomToHU(tags=["bcolzall:3d"]),
    Augment3D(
        tags=["bcolzall:3d"],
        output_shape = nn_input_shape,
        norm_patch_shape=norm_patch_shape,
        interp_order=1,
        mode="constant"),
    ZMUV("bcolzall:3d", bias=-648.59027, std=679.21021)
]


#####################
#     training      #
#####################

"This is the train dataloader. We will train until this one stops loading data."
"You can set the number of epochs, the datasets and if you want it multiprocessed"
n_epochs = 100
training_data = BcolzAllDataLoader(
    sets=TRAINING,
    epochs=n_epochs,
    preprocessors=preprocessors,
    multiprocess=multiprocessing_on,
    crash_on_exception=True)

"Schedule the reducing of the learning rate. On indexing with the number of epochs, it should return a value for the learning rate." 
lr = 0.000001
lr_min = lr/1000.
lr_decay = 0.95
learning_rate_schedule = {}
for i in range(n_epochs):
    lr_ = lr*(lr_decay**i)
    if lr_ < lr_min: break
    learning_rate_schedule[i] = lr_

# print learning_rate_schedule

"The function to build updates."
build_updates = lasagne.updates.nesterov_momentum


#####################
#    validation     #
#####################
"We do our validation after every x epochs of training"
epochs_per_validation = 2.0

"Which data do we want to validate on. We will run all validation objectives on each validation data set."
validation_data = {
    "validation set": BcolzAllDataLoader(sets=VALIDATION,
        epochs=1,
        preprocessors=preprocessors_valid,
        process_last_chunk=True,
        multiprocess=multiprocessing_on,
        crash_on_exception=True),
    "training set":  BcolzAllDataLoader(sets=TRAINING,
        epochs=0.01,
        preprocessors=preprocessors_valid,
        process_last_chunk=True,
        multiprocess=multiprocessing_on,
        crash_on_exception=True)
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
    obj = CrossEntropyObjective(interface_layers["outputs"], target_name="bcolzall")
    return {
        "train":{
            "objective": obj,
        },
        "validate":{
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
    l_in = lasagne.layers.InputLayer(shape=(None,)+nn_input_shape)

    l = lasagne.layers.DimshuffleLayer(l_in, pattern=(0, 3, "x", 2, 1))

    l = lasagne.layers.ReshapeLayer(l, (-1, 1, nn_input_shape[0], nn_input_shape[1]))


    # n = 16
    # l = conv2d(l, n, filter_size=7, stride=2)
    # l = wn(l)
    # # 256
    # n *= 2
    # l = conv2d(l, n)
    # l = wn(l)
    # l = conv2d(l, n)
    # l = wn(l)
    # l = conv2d(l, n)
    # l = wn(l)
    # l = max_pool2d(l)
    
    # 128
    n = 128
    l = conv2d(l, n)
    l = wn(l)
    l = conv2d(l, n)
    l = wn(l)
    l = conv2d(l, n)
    l = wn(l)
    l = conv2d(l, n)
    l = wn(l)
    l = max_pool2d(l, pool_size=16)
    #64
    # n *= 2
    # l = conv2d(l, n)
    # l = wn(l)
    # l = conv2d(l, n)
    # l = wn(l)
    # l = max_pool2d(l)
    # # 32
    # n *= 2
    # l = conv2d(l, n)
    # l = wn(l)
    # l = conv2d(l, n)
    # l = wn(l)
    # l = max_pool2d(l)
    # # 16
    # # n *= 2
    # l = conv2d(l, n)
    # l = wn(l)
    # l = conv2d(l, n)
    # l = wn(l)
    # l = max_pool2d(l)
    # # 8
    # l = conv2d(l, n)
    # l = wn(l)
    # l = conv2d(l, n)
    # l = wn(l)
    # l = max_pool2d(l)

    n_features = np.prod(l.output_shape[-3:])
    l = lasagne.layers.ReshapeLayer(l, (-1,  nn_input_shape[2], n_features))
    l = lasagne.layers.FeaturePoolLayer(l, nn_input_shape[2], axis=1)
    # l = lasagne.layers.ReshapeLayer(l, (-1, n_features))

    n *= 8
    l = dense(l, n)
    l = wn(l)
    # l = drop(l)
    l = dense(l, n)
    l = wn(l)
    # l = drop(l)

    l = lasagne.layers.DenseLayer(l,
                                 num_units=1,
                                 W=lasagne.init.Constant(0.),
                                 b=lasagne.init.Constant(-np.log(1./0.25-1.)),
                                 nonlinearity=lasagne.nonlinearities.sigmoid)
    l_out = lasagne.layers.reshape(l, shape=(-1,))

    return {
        "inputs":{
            "bcolzall:3d": l_in,
        },
        "outputs": {
            "predicted_probability": l_out
        },
    }
