from application.preprocessors.in_the_middle import Put_In_The_Middle
from configurations.default import *

import lasagne
import theano.tensor as T
import numpy as np

from application.objectives import CrossEntropyObjective
from application.data import PatientDataLoader
from interfaces.data_loader import VALIDATION, TRAINING, TEST, TRAIN
from deep_learning.deep_learning_layers import ConvolutionLayer, PoolLayer

#####################
#   running speed   #
#####################
from interfaces.preprocess import NormalizeInput

batch_size = 3
batches_per_chunk = 4
restart_from_save = False
save_every_chunks = 1


#####################
#   preprocessing   #
#####################

preprocessors = [
    Put_In_The_Middle(tag="3d",output_shape=(256,512,512))
]


#####################
#     training      #
#####################

training_data = PatientDataLoader(sets=TRAINING,
                                  epochs=10.0,
                                  preprocessors=preprocessors,
                                 multiprocess=False,
                                 crash_on_exception=True,
                                  )

learning_rate_schedule = {
    0.0: 0.001,
    9.0: 0.0001,
}
build_updates = lasagne.updates.adam


#####################
#    validation     #
#####################

epochs_per_validation = 1.0
validation_data = {
    "validation set": PatientDataLoader(sets=VALIDATION,
                                        epochs=1,
                                        preprocessors=preprocessors,
                                        process_last_chunk=True,
                                 multiprocess=False,
                                 crash_on_exception=True,
                                        ),
    "training set":  PatientDataLoader(sets=TRAINING,
                                        epochs=0.01,
                                        preprocessors=preprocessors,
                                        process_last_chunk=True,
                                 multiprocess=False,
                                 crash_on_exception=True,
                                        ),
    }


#####################
#      testing      #
#####################

test_data = PatientDataLoader(sets=[TEST, VALIDATION, TRAINING],
                              epochs=1,
                              preprocessors=preprocessors,
                              process_last_chunk=True
                              )


#####################
#     debugging     #
#####################

def build_objectives(interface_layers):
    obj = CrossEntropyObjective(interface_layers["outputs"], target_name="dsb3")
    return {
        "train":{
            "objective": obj,
        },
        "validate":{
            "objective": obj,
        }
    }

def build_model():

    #################
    # Regular model #
    #################

    l0 = lasagne.layers.InputLayer(shape=(None,256,512,512))

    l_dense = lasagne.layers.DenseLayer(l0,
                                         num_units=1,
                                         W=lasagne.init.Constant(0.0),
                                         b=None,
                                         nonlinearity=lasagne.nonlinearities.sigmoid)
    l_output = lasagne.layers.reshape(l_dense, shape=(-1,))

    return {
        "inputs":{
            "dsb3:3d": l0,
        },
        "outputs": {
            "predicted_probability": l_output
        },
    }