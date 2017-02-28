from functools import partial
from lasagne.layers import dnn
from application.luna import LunaDataLoader, OnlyPositiveLunaDataLoader

import lasagne
from configurations.default import *
from application.objectives import CrossEntropyObjective
from application.preprocessors.augment_only_positive_candidates import AugmentOnlyPositiveCandidates
from deep_learning.upscale import Upscale3DLayer
from interfaces.data_loader import VALIDATION, TRAINING, TEST, TRAIN
from deep_learning.deep_learning_layers import ConvolutionLayer, PoolLayer
from deep_learning.nn_lung import Hu2normHULayer, ZMUVLayer


#####################
#   running speed   #
#####################
from interfaces.preprocess import NormalizeInput, ZMUV

"This is the number of samples in each batch"
batch_size = 32
"This is the number of batches in each chunk. Computation speeds up if this is as big as possible." \
"However, when too big, the GPU will run out of memory"
batches_per_chunk = 1
"Reload the parameters from last time and continue, or start anew when you run this config file again"
restart_from_save = False
"After how many chunks should you save parameters. Keep this number high for better performance. It will always store at end anyway"
save_every_chunks = 50


#####################
#   preprocessing   #
#####################

AUGMENTATION_PARAMETERS = {
    "scale": [1, 1, 1],  # factor
    "rotation": [180, 180, 180],  # degrees (from -180 to 180)
    "shear": [0, 0, 0],  # degrees
    "translation": [3, 3, 3],  # mms (from -128 to 128)
    "reflection": [0, 0, 0] #Bernoulli p
}

#there was a bug in luna_direct_x23.py, so I took 32 now, but we should try out 48
IMAGE_SIZE = 32
num_epochs=100

"Put in here the preprocessors for your data." \
"They will be run consequently on the datadict of the dataloader in the order of your list."
preprocessors = {
    "train":
    [   AugmentOnlyPositiveCandidates(train_valid='train', tags=["luna:3d", "luna:target"],
               output_shape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE),  # in pixels
               norm_patch_shape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE),  # in mms
               augmentation_params=AUGMENTATION_PARAMETERS
               ),
        #ZMUV("luna:3d", bias =  -648.59027, std = 679.21021),
    ],
    "valid":
    [   AugmentOnlyPositiveCandidates(train_valid='valid', tags=["luna:3d", "luna:target"],
               output_shape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE),  # in pixels
               norm_patch_shape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE),  # in mms
               augmentation_params=AUGMENTATION_PARAMETERS
               ),
        #ZMUV("luna:3d", bias =  -648.59027, std = 679.21021),
    ]
}


#####################
#     training      #
#####################
"This is the train dataloader. We will train until this one stops loading data."
"You can set the number of epochs, the datasets and if you want it multiprocessed"
training_data = LunaDataLoader(
    only_positive=True,
    sets=TRAINING,
    epochs=num_epochs,
    preprocessors=preprocessors['train'],
    multiprocess=True,
    crash_on_exception=True,
)

"Schedule the reducing of the learning rate. On indexing with the number of epochs, it should return a value for the learning rate."
learning_rate_schedule = {
    0: 5e-4,
    int(num_epochs * 0.5): 1e-4,
    int(num_epochs * 0.6): 5e-5,
    int(num_epochs * 0.7): 2e-5,
    int(num_epochs * 0.8): 1e-5,
    int(num_epochs * 0.9): 5e-6
}


"The function to build updates."
build_updates = lasagne.updates.adam


#####################
#    validation     #
#####################
"We do our validation after every x epochs of training, and at the end"
epochs_per_validation = 1

"Which data do we want to validate on. We will run all validation objectives on each validation data set."
validation_data = {
    "validation set": LunaDataLoader(
        sets=VALIDATION,
        only_positive=True,
        epochs=1,
        preprocessors=preprocessors['valid'],
        process_last_chunk=True,
        multiprocess=True,
        crash_on_exception=True,
    ),
    "training set": LunaDataLoader(
        sets=TRAINING,
        only_positive=True,
        epochs=0.01,
        #should be valid because we always want to check the same patches
        preprocessors=preprocessors['valid'],
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
    # TODO I am not so sure about target_name
    obj = CrossEntropyObjective(interface_layers["outputs"], target_name="luna")
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
                 W=lasagne.init.Orthogonal(),
                 b=lasagne.init.Constant(0.01),
                 nonlinearity=lasagne.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

drop = lasagne.layers.DropoutLayer

bn = lasagne.layers.batch_norm

dense = partial(lasagne.layers.DenseLayer,
    W=lasagne.init.Orthogonal('relu'),
    b=lasagne.init.Constant(0.0),
    nonlinearity=lasagne.nonlinearities.rectify)


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

    l = lasagne.layers.ElemwiseSumLayer([l,lin])

    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)

    return l

def inrn_v2_red(lin):
    #We want to reduce our total volume /4

    den = 16
    nom2 = 4
    nom3 = 5
    nom4 = 7

    ins = lin.output_shape[1]

    l1 = max_pool3d(lin)

    l2 = conv3d(lin, ins//den*nom2, filter_size=3, stride=2)

    l3 = conv3d(lin, ins//den*nom2, filter_size=1)
    l3 = conv3d(l3, ins//den*nom3, filter_size=3, stride=2)

    l4 = conv3d(lin, ins//den*nom2, filter_size=1)
    l4 = conv3d(l4, ins//den*nom3, filter_size=3)
    l4 = conv3d(l4, ins//den*nom4, filter_size=3, stride=2)

    l = lasagne.layers.ConcatLayer([l1, l2, l3, l4])

    return l

"Here we build a model. The model returns a dict with the requested inputs for each layer:" \
"And with the outputs it generates. You may generate multiple outputs (for analysis or for some other objectives, etc)" \
"Unused outputs don't cost in performance"

def build_model(image_size=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE)):



    l_in = lasagne.layers.InputLayer(shape=(None,)+image_size)
    l_norm = Hu2normHULayer(l_in, min_hu=-1000,max_hu=400)
    l_norm = ZMUVLayer(l_norm, mean=0.36, std=0.31)
    l0 = lasagne.layers.DimshuffleLayer(l_norm, pattern=[0,'x',1,2,3])
    

    l = conv3d(l0, 64)
    l = inrn_v2_red(l)
    l = inrn_v2(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)
    l = inrn_v2(l)

    l = dense(drop(l), 128)

    # this is a different way to output compared to luna_direct_x23 config
    l = lasagne.layers.DenseLayer(l, num_units=1,
                                 W=lasagne.init.Constant(0.),
                                 nonlinearity=lasagne.nonlinearities.sigmoid)
    
    
    l_out = lasagne.layers.reshape(l, shape=(-1,))
    
    return {
        "inputs":{
            "luna:3d": l_in,
        },
        "outputs": {
            "predicted_probability": l_out
        },
    }
