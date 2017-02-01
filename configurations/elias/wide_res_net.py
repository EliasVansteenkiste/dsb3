from application.preprocessors.in_the_middle import PutInTheMiddle
from configurations.default import *

import lasagne
from lasagne.layers import dnn
from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv3DDNNLayer
from lasagne.layers.dnn import Pool3DDNNLayer
from lasagne.layers.dnn import Conv3DDNNLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import DimshuffleLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import reshape
from lasagne.layers import DenseLayer
from lasagne.layers import batch_norm

from lasagne.nonlinearities import rectify, softmax, identity
from lasagne.init import Orthogonal, HeNormal, GlorotNormal
import theano.tensor as T
import numpy as np
from functools import partial

from application.objectives import CrossEntropyObjective
#from application.data import PatientDataLoader
from application.bcolz_all_data import BcolzAllDataLoader
from interfaces.data_loader import VALIDATION, TRAINING, TEST, TRAIN
from application.preprocessors.augmentation_3d import Augment3D
from application.preprocessors.normalize_scales import DefaultNormalizer

he_norm = HeNormal(gain='relu')

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
lr = 0.0001 
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
    nonlinearity=None)

nl = partial(NonlinearityLayer,
    nonlinearity=lasagne.nonlinearities.leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer, pool_size=2)

dense = partial(lasagne.layers.DenseLayer,
    W=lasagne.init.Orthogonal('relu'),
    b=lasagne.init.Constant(0.0),
    nonlinearity=lasagne.nonlinearities.leaky_rectify)

drop = lasagne.layers.DropoutLayer

bn = lasagne.layers.batch_norm


# create a residual learning building block with two stacked 3x3 convlayers as in paper
def residual_block(l, increase_dim=False, projection=True, first=False, filters=16):
    if increase_dim:
        first_stride = 2
    else:
        first_stride = 1

    if first:
        # hacky solution to keep layers correct
        bn_pre_relu = l
    else:
        # contains the BN -> ReLU portion, steps 1 to 2
        bn_pre_conv = BatchNormLayer(l)
        bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

    # contains the weight -> BN -> ReLU portion, steps 3 to 5
    conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=3, stride=first_stride, nonlinearity=rectify, pad='same', W=he_norm))

    dropout = DropoutLayer(conv_1, p=0.3)

    # contains the last weight portion, step 6
    conv_2 = ConvLayer(conv_1, num_filters=filters, filter_size=3, stride=1, nonlinearity=None, pad='same', W=he_norm)

    # add shortcut connections
    if increase_dim:
        # projection shortcut, as option B in paper
        projection = ConvLayer(bn_pre_relu, num_filters=filters, filter_size=1, stride=2, nonlinearity=None, pad='same', b=None)
        block = ElemwiseSumLayer([conv_2, projection])

    elif first:
        # projection shortcut, as option B in paper
        projection = ConvLayer(l, num_filters=filters, filter_size=1, stride=1, nonlinearity=None, pad='same', b=None)
        block = ElemwiseSumLayer([conv_2, projection])

    else:
        block = ElemwiseSumLayer([conv_2, l])

    return block



def build_model():
    '''
    Adapted from https://github.com/Lasagne/Recipes/tree/master/papers/deep_residual_learning.
    Tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    And 'Wide Residual Networks', Sergey Zagoruyko, Nikos Komodakis 2016 (http://arxiv.org/pdf/1605.07146v1.pdf)
    Depth = 6n + 2
    '''

    n=6
    k=4
    n_filters = {0:16, 1:16*k, 2:32*k, 3:64*k}

    l_in = InputLayer(shape=(None,)+nn_input_shape)
    l = DimshuffleLayer(l_in, pattern=(0, 'x', 1, 2, 3))

    # first layer, output is 16 x 64 x 64
    l = batch_norm(ConvLayer(l, num_filters=n_filters[0], filter_size=3, stride=1, nonlinearity=rectify, pad='same', W=he_norm))

    # first stack of residual blocks, output is 32 x 64 x 64
    l = residual_block(l, first=True, filters=n_filters[1])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[1])

    # second stack of residual blocks, output is 64 x 32 x 32
    l = residual_block(l, increase_dim=True, filters=n_filters[2])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[2])

    # third stack of residual blocks, output is 128 x 16 x 16
    l = residual_block(l, increase_dim=True, filters=n_filters[3])
    for _ in range(1,(n+2)):
        l = residual_block(l, filters=n_filters[3])

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)


    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    sigm = DenseLayer(avg_pool, num_units=1,
                                 W=lasagne.init.Constant(0.0),
                                 b=None,
                                 nonlinearity=lasagne.nonlinearities.sigmoid)

    l_out = reshape(sigm, shape=(-1,))

    return {
        "inputs":{
            "bcolzall:3d": l_in,
        },
        "outputs": {
            "predicted_probability": l_out
        },
    }
