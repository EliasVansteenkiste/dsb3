from functools import partial
from lasagne.layers import dnn
from application.luna import LunaDataLoader, OnlyPositiveLunaDataLoader

import lasagne
from configurations.default import *
from application.objectives import CrossEntropyObjective, WeightedSegmentationCrossEntropyObjective, \
    JaccardIndexObjective, SoerensonDiceCoefficientObjective, RecallObjective, PrecisionObjective, ClippedFObjective
from application.preprocessors.augment_only_positive import AugmentOnlyPositive
from deep_learning.upscale import Upscale3DLayer
from interfaces.data_loader import VALIDATION, TRAINING, TEST, TRAIN
from deep_learning.deep_learning_layers import ConvolutionLayer, PoolLayer
from deep_learning.nn_lung import Hu2normHULayer


#####################
#   running speed   #
#####################
from interfaces.preprocess import NormalizeInput, ZMUV

"This is the number of samples in each batch"
batch_size = 4
"This is the number of batches in each chunk. Computation speeds up if this is as big as possible." \
"However, when too big, the GPU will run out of memory"
batches_per_chunk = 8
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
    "translation": [16, 16, 16],  # mms (from -128 to 128)
    "reflection": [0.5, 0.5, 0.5] #Bernoulli p
}

IMAGE_SIZE = 64
num_epochs=100

"Put in here the preprocessors for your data." \
"They will be run consequently on the datadict of the dataloader in the order of your list."
preprocessors = {
    "train":
    [   AugmentOnlyPositive(train_valid='train', tags=["luna:3d", "luna:segmentation"],
               output_shape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE),  # in pixels
               norm_patch_shape=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE),  # in mms
               augmentation_params=AUGMENTATION_PARAMETERS
               ),
        #ZMUV("luna:3d", bias =  -648.59027, std = 679.21021),
    ],
    "valid":
    [   AugmentOnlyPositive(train_valid='valid', tags=["luna:3d", "luna:segmentation"],
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
    0: 1e-5,
    int(num_epochs * 0.4): 5e-6,
    int(num_epochs * 0.5): 3e-6,
    int(num_epochs * 0.6): 2e-6,
    int(num_epochs * 0.85): 1e-6,
    int(num_epochs * 0.95): 5e-7
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
    obj_weighted = WeightedSegmentationCrossEntropyObjective(
        classweights=[10000, 1],
        input_layer=interface_layers["outputs"]["predicted_segmentation"],
        target_name="luna:segmentation",
    )

    obj_jaccard = JaccardIndexObjective(
        smooth=1e-5,
        input_layer=interface_layers["outputs"]["predicted_segmentation"],
        target_name="luna:segmentation",
    )

    obj_dice = SoerensonDiceCoefficientObjective(
        input_layer=interface_layers["outputs"]["predicted_segmentation"],
        target_name="luna:segmentation",
    )

    obj_precision = PrecisionObjective(
        smooth=1e-5,
        input_layer=interface_layers["outputs"]["predicted_segmentation"],
        target_name="luna:segmentation",
    )

    obj_recall = RecallObjective(
        smooth=1e-5,
        input_layer=interface_layers["outputs"]["predicted_segmentation"],
        target_name="luna:segmentation",
    )

    obj_custom = ClippedFObjective(
        smooth=1e-5,
        recall_weight = 1./0.95,
        precision_weight = 1./0.3,
        input_layer=interface_layers["outputs"]["predicted_segmentation"],
        target_name="luna:segmentation",
    )

    return {
        "train":{
            "objective": obj_dice,
            "jaccard": obj_jaccard,
            "Dice": obj_dice,
            "precision": obj_precision,
            "recall": obj_recall,
        },
        "validate":{
            "objective": obj_dice,
            "jaccard": obj_jaccard,
            "Dice": obj_dice,
            "precision": obj_precision,
            "recall": obj_recall,
        }
    }

#################
# Regular model #
#################


"For ease of working, we predefine some layers with their parameters"
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='valid',
                 W=lasagne.init.Orthogonal('relu'),
                 b=lasagne.init.Constant(0.0),
                 nonlinearity=lasagne.nonlinearities.identity)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

"Here we build a model. The model returns a dict with the requested inputs for each layer:" \
"And with the outputs it generates. You may generate multiple outputs (for analysis or for some other objectives, etc)" \
"Unused outputs don't cost in performance"

def build_model(image_size=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE)):

    l_in = lasagne.layers.InputLayer(shape=(None,)+image_size)
    l_norm = Hu2normHULayer(l_in, min_hu=-1000,max_hu=400)
    l0 = lasagne.layers.DimshuffleLayer(l_norm, pattern=[0,'x',1,2,3])
    

    net = {}
    base_n_filters = 128
    net['contr_1_1'] = conv3d(l0, base_n_filters)
    net['contr_1_1'] = lasagne.layers.ParametricRectifierLayer(net['contr_1_1'])
    net['contr_1_2'] = conv3d(net['contr_1_1'], base_n_filters)
    net['contr_1_2'] = lasagne.layers.ParametricRectifierLayer(net['contr_1_2'])
    net['contr_1_3'] = conv3d(net['contr_1_2'], base_n_filters)
    net['contr_1_3'] = lasagne.layers.ParametricRectifierLayer(net['contr_1_3'])
    net['pool1'] = max_pool3d(net['contr_1_3'])

    net['encode_1'] = conv3d(net['pool1'], base_n_filters)
    net['encode_1'] = lasagne.layers.ParametricRectifierLayer(net['encode_1'])
    net['encode_2'] = conv3d(net['encode_1'], base_n_filters)
    net['encode_2'] = lasagne.layers.ParametricRectifierLayer(net['encode_2'])
    net['encode_3'] = conv3d(net['encode_2'], base_n_filters)
    net['encode_3'] = lasagne.layers.ParametricRectifierLayer(net['encode_3'])
    net['encode_4'] = conv3d(net['encode_3'], base_n_filters)
    net['encode_4'] = lasagne.layers.ParametricRectifierLayer(net['encode_4'])
    
    net['dropout_1'] = lasagne.layers.DropoutLayer(net['encode_4'])
    
    net['upscale1'] = lasagne.layers.Upscale3DLayer(net['dropout_1'], 2)

    net['concat1'] = lasagne.layers.ConcatLayer([net['upscale1'], net['contr_1_3']],
                                           cropping=(None, None, "center", "center", "center"))
    net['expand_1_1'] = conv3d(net['concat1'], 2 * base_n_filters)
    net['expand_1_1'] = lasagne.layers.ParametricRectifierLayer(net['expand_1_1'])
    net['expand_1_2'] = conv3d(net['expand_1_1'], 2 * base_n_filters)
    net['expand_1_2'] = lasagne.layers.ParametricRectifierLayer(net['expand_1_2'])
    net['expand_1_3'] = conv3d(net['expand_1_2'], base_n_filters)
    net['expand_1_3'] = lasagne.layers.ParametricRectifierLayer(net['expand_1_3'])

    net['output_segmentation'] = dnn.Conv3DDNNLayer(net['expand_1_3'], num_filters=1,
                               filter_size=1,
                               nonlinearity=lasagne.nonlinearities.sigmoid)

    l_out = lasagne.layers.SliceLayer(net['output_segmentation'], indices=0, axis=1)
    
    
    
    return {
        "inputs":{
            "luna:3d": l_in,
        },
        "outputs": {
            "predicted_segmentation": l_out
        },
    }
