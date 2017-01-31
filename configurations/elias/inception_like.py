from application.preprocessors.in_the_middle import PutInTheMiddle
from configurations.default import *

import lasagne
from lasagne.layers import dnn, NonlinearityLayer, ElemwiseSumLayer, GlobalPoolLayer, ConcatLayer, DenseLayer, reshape, InputLayer, DimshuffleLayer
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

nl = partial(NonlinearityLayer,
    nonlinearity=lasagne.nonlinearities.leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer, pool_size=2)
mp3d = dnn.MaxPool3DDNNLayer

dense = partial(lasagne.layers.DenseLayer,
    W=lasagne.init.Orthogonal('relu'),
    b=lasagne.init.Constant(0.0),
    nonlinearity=lasagne.nonlinearities.leaky_rectify)

drop = lasagne.layers.DropoutLayer

bn = lasagne.layers.batch_norm

Pool3DLayer = dnn.Pool3DDNNLayer


conv3d = partial(dnn.Conv3DDNNLayer,
    pad='same',
    W=lasagne.init.HeNormal('relu'),
    b=lasagne.init.Constant(0.0),
    nonlinearity=None)

def bc(input_layer, **kwargs):
    l = conv3d(input_layer, **kwargs)
    l = nl(bn(l, epsilon=0.001))
    return l


def inceptionA(input_layer, nfilt):
    # Corresponds to a modified version of figure 5 in the paper
    l1 = bc(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bc(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bc(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

    l3 = bc(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bc(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3 = bc(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

    l4 = Pool3DLayer(input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bc(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionB(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bc(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

    l2 = bc(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bc(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    l2 = bc(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

    l3 = Pool3DLayer(input_layer, pool_size=3, stride=2, pad=1)

    print 'inceptionB'
    print l1.output_shape
    print l2.output_shape
    print l3.output_shape

    return ConcatLayer([l1, l2, l3])


def inceptionC(input_layer, nfilt):
    # Corresponds to figure 6 in the paper
    l1 = bc(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bc(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bc(l2, num_filters=nfilt[1][1], filter_size=(1, 1, 7), pad=(0, 0, 3))
    l2 = bc(l2, num_filters=nfilt[1][2], filter_size=(1, 7, 1), pad=(0, 3, 0))
    l2 = bc(l2, num_filters=nfilt[1][3], filter_size=(7, 1, 1), pad=(3, 0, 0))

    l3 = bc(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l2 = bc(l2, num_filters=nfilt[2][1], filter_size=(1, 1, 7), pad=(0, 0, 3))
    l2 = bc(l2, num_filters=nfilt[2][2], filter_size=(1, 7, 1), pad=(0, 3, 0))
    l2 = bc(l2, num_filters=nfilt[2][3], filter_size=(7, 1, 1), pad=(3, 0, 0))
    l2 = bc(l2, num_filters=nfilt[2][4], filter_size=(1, 1, 7), pad=(0, 0, 3))
    l2 = bc(l2, num_filters=nfilt[2][5], filter_size=(1, 7, 1), pad=(0, 3, 0))
    l2 = bc(l2, num_filters=nfilt[2][6], filter_size=(7, 1, 1), pad=(3, 0, 0))

    l4 = Pool3DLayer(input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bc(l4, num_filters=nfilt[3][0], filter_size=1)

    print 'inceptionC'
    print l1.output_shape
    print l2.output_shape
    print l3.output_shape
    print l4.output_shape

    return ConcatLayer([l1, l2, l3, l4])


def inceptionD(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bc(input_layer, num_filters=nfilt[0][0], filter_size=1)
    l1 = bc(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

    l2 = bc(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bc(l2, num_filters=nfilt[1][1], filter_size=(1, 1, 7), pad=(0, 0, 3))
    l2 = bc(l2, num_filters=nfilt[1][2], filter_size=(1, 7, 1), pad=(0, 3, 0))
    l2 = bc(l2, num_filters=nfilt[1][3], filter_size=(7, 1, 1), pad=(3, 0, 0))
    l2 = bc(l2, num_filters=nfilt[1][4], filter_size=3, stride=2)

    l3 = Pool3DLayer(input_layer, pool_size=3, stride=2, pad=1)

    return ConcatLayer([l1, l2, l3])


def inceptionE(input_layer, nfilt, pool_mode):
    # Corresponds to figure 7 in the paper
    l1 = bc(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bc(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2a = bc(l2, num_filters=nfilt[1][1], filter_size=(1, 1, 3), pad=(0, 0, 1))
    l2b = bc(l2, num_filters=nfilt[1][2], filter_size=(1, 3, 1), pad=(0, 1, 0))
    l2c = bc(l2, num_filters=nfilt[1][3], filter_size=(3, 1, 1), pad=(1, 0, 0))

    l3 = bc(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bc(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3a = bc(l3, num_filters=nfilt[2][2], filter_size=(1, 1, 3), pad=(0, 0, 1))
    l3b = bc(l3, num_filters=nfilt[2][3], filter_size=(1, 3, 1), pad=(0, 1, 0))
    l3c = bc(l3, num_filters=nfilt[2][4], filter_size=(3, 1, 1), pad=(1, 0, 0))

    l4 = Pool3DLayer(input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

    l4 = bc(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2a, l2b, l2c, l3a, l3b, l3c, l4])


def build_model():
    net = {}

    net['input'] = InputLayer(shape=(None,)+nn_input_shape)
    net['reshuffle'] = DimshuffleLayer(net['input'], pattern=(0, 'x', 1, 2, 3))

    net['conv'] = bc(net['reshuffle'],num_filters=32, filter_size=3, stride=2)
    net['conv_1'] = bc(net['conv'], num_filters=32, filter_size=3)
    net['conv_2'] = bc(net['conv_1'],num_filters=64, filter_size=3, pad=1)
    net['pool'] = Pool3DLayer(net['conv_2'], pool_size=3, stride=2, mode='max')

    # net['conv_3'] = bc(net['pool'], num_filters=80, filter_size=1)

    # net['conv_4'] = bc(net['conv_3'], num_filters=192, filter_size=3)

    # net['pool_1'] = Pool3DLayer(net['conv_4'], pool_size=3, stride=2, mode='max')
    
    # I divided all the number of filters by 2
    net['mixed/join'] = inceptionA(net['pool'], nfilt=((32,), (24, 32), (32, 48, 48), (16,)))
    net['mixed_1/join'] = inceptionA(net['mixed/join'], nfilt=((32,), (24, 32), (32, 48, 48), (32,)))
    net['mixed_2/join'] = inceptionA(net['mixed_1/join'], nfilt=((32,), (24, 32), (32, 48, 48), (32,)))
    
    # I divided all the number of filters by 2
    net['mixed_3/join'] = inceptionB(net['mixed_2/join'], nfilt=((192,), (32, 48, 48)))

    # I divided all the number of filters by 4
    net['mixed_4/join'] = inceptionC(net['mixed_3/join'], nfilt=((48,), (32, 32, 32, 48), (32, 32, 32, 32, 32, 32, 48), (48,)))
    net['mixed_5/join'] = inceptionC(net['mixed_4/join'], nfilt=((48,), (40, 40, 40, 48), (40, 40, 40, 40, 40, 40, 48), (48,)))
    net['mixed_6/join'] = inceptionC(net['mixed_5/join'], nfilt=((48,), (40, 40, 40, 48), (40, 40, 40, 40, 40, 40, 48), (48,)))
    net['mixed_7/join'] = inceptionC(net['mixed_6/join'], nfilt=((48,), (48, 48, 40, 48), (48, 48, 48, 48, 48, 48, 48), (48,)))
    
    net['mixed_8/join'] = inceptionD(net['mixed_7/join'], nfilt=((48, 80), (48, 48, 48, 48, 48)))

    net['mixed_9/join'] = inceptionE(net['mixed_8/join'], nfilt=((80,), (96, 96, 96, 96), (112, 96, 96, 96, 96), (48,)),pool_mode='average_exc_pad')

    net['mixed_10/join'] = inceptionE(net['mixed_9/join'], nfilt=((80,), (96, 96, 96, 96), (112, 96, 96, 96, 96), (48,)),pool_mode='max')

    net['pool3'] = GlobalPoolLayer(net['mixed_10/join'])

    net['sigmoid'] = DenseLayer(net['pool3'], num_units=1, W=lasagne.init.Constant(0.0), b=None, nonlinearity=lasagne.nonlinearities.sigmoid)

    net['output'] = reshape(net['sigmoid'], shape=(-1,))

    return {
        "inputs":{
            "bcolzall:3d": net['input'],
        },
        "outputs": {
            "predicted_probability": net['output']
        },
    }

