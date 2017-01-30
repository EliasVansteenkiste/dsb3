from application.preprocessors.in_the_middle import PutInTheMiddle
from configurations.default import *

import lasagne
from lasagne.layers import dnn, NonlinearityLayer, ElemwiseSumLayer, GlobalPoolLayer
import theano.tensor as T
import numpy as np
from functools import partial

from application.objectives import CrossEntropyObjective
#from application.data import PatientDataLoader
from application.bcolz_all_data import BcolzAllDataLoader
from interfaces.data_loader import VALIDATION, TRAINING, TEST, TRAIN
from application.preprocessors.augmentation_3d import Augment3D
from application.preprocessors.normalize_scales import DefaultNormalizer
from deep_learning.deep_learning_layers import ConvolutionLayer, PoolLayer

#####################
#   running speed   #
#####################
from interfaces.preprocess import NormalizeInput

"This is the number of samples in each batch"
batch_size = 16
"This is the number of batches in each chunk. Computation speeds up if this is as big as possible." \
"However, when too big, the GPU will run out of memory"
batches_per_chunk = 1
"Reload the parameters from last time and continue, or start anew when you run this config file again"
restart_from_save = False
"After how many chunks should you save parameters. Keep this number high for better performance. It will always store at end anyway"
save_every_chunks = 10


#####################
#   preprocessing   #
#####################

"Put in here the preprocessors for your data." \
"They will be run consequently on the datadict of the dataloader in the order of your list."
nn_input_shape = (64, 128, 128)
preprocessors = [
    Augment3D(
        tags=["bcolzall:3d"],
        output_shape = nn_input_shape,
        norm_patch_shape=(320, 340, 340),
        augmentation_params={
            "scale": [1, 1, 1],  # factor
            "rotation": [3, 3, 3],  # degrees
            "shear": [0, 0, 0],  # degrees
            "translation": [20, 20, 20],  # mm
            "reflection": [0, 0, 0]}, #Bernoulli p
        interp_order=1),
    DefaultNormalizer(tags=["bcolzall:3d"])
]


#####################
#     training      #
#####################

"This is the train dataloader. We will train until this one stops loading data."
"You can set the number of epochs, the datasets and if you want it multiprocessed"
training_data = BcolzAllDataLoader(
    sets=TRAINING,
    epochs=50, #infinite
    preprocessors=preprocessors,
    multiprocess=True,
    crash_on_exception=True)

"Schedule the reducing of the learning rate. On indexing with the number of epochs, it should return a value for the learning rate." 
lr = 0.0002 
lr_decay = 0.9 # per epoch
learning_rate_schedule = {}
for i in range(100):
    learning_rate_schedule[float(i)] = lr*(lr_decay**i)

#print learning_rate_schedule

"The function to build updates."
build_updates = lasagne.updates.adam


#####################
#    validation     #
#####################
"We do our validation after every x epochs of training"
epochs_per_validation = 1.0

"Which data do we want to validate on. We will run all validation objectives on each validation data set."
validation_data = {
    "validation set": BcolzAllDataLoader(sets=VALIDATION,
        epochs=1,
        preprocessors=preprocessors,
        process_last_chunk=True,
        multiprocess=True,
        crash_on_exception=True),
 #   "training set": None
    "training set":  BcolzAllDataLoader(sets=TRAINING,
                                         epochs=0.2,
                                         preprocessors=preprocessors,
                                         process_last_chunk=True,
                                  multiprocess=True,
                                  crash_on_exception=True,
                                         ),
    }


#####################
#      testing      #
#####################
"This is the data which will be used for testing."
test_data = None


#####################
#     debugging     #
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
    W=lasagne.init.HeNormal('relu'),
    b=lasagne.init.Constant(0.0),
    nonlinearity=lasagne.nonlinearities.very_leaky_rectify)

nl = partial(NonlinearityLayer,
    nonlinearity=lasagne.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer, pool_size=2)

dropout = lasagne.layers.dropout

dense = partial(lasagne.layers.DenseLayer,
    W=lasagne.init.Orthogonal('relu'),
    b=lasagne.init.Constant(0.0),
    nonlinearity=lasagne.nonlinearities.leaky_rectify)


bn = lasagne.layers.batch_norm




"Here we build a model. The model returns a dict with the requested inputs for each layer:" \
"And with the outputs it generates. You may generate multiple outputs (for analysis or for some other objectives, etc)" \
"Unused outputs don't cost in performance"
def build_model():
    l_in = lasagne.layers.InputLayer(shape=(None,)+nn_input_shape)

    l = lasagne.layers.DimshuffleLayer(l_in, pattern=(0, 'x', 1, 2, 3))

    l = nl(bn(conv3d(l, num_filters=32, filter_size=5, stride=2)))
    l = nl(bn(conv3d(l, num_filters=32)))
    l = max_pool3d(l)
    l = dropout(l)

    l = nl(bn(conv3d(l, num_filters=64)))
    l = nl(bn(conv3d(l, num_filters=64)))
    l = max_pool3d(l)
    l = dropout(l)

    l = nl(bn(conv3d(l, num_filters=64)))
    l = nl(bn(conv3d(l, num_filters=64)))
    l = max_pool3d(l)
    l = dropout(l)


    l = GlobalPoolLayer(l)

    l = lasagne.layers.DenseLayer(l,
                                 num_units=1,
                                 W=lasagne.init.Constant(0.0),
                                 b=None,
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
