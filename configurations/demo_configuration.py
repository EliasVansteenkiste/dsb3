from application.preprocessors.in_the_middle import PutInTheMiddle
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

"This is the number of samples in each batch"
batch_size = 1
"This is the number of batches in each chunk. Computation speeds up if this is as big as possible." \
"However, when too big, the GPU will run out of memory"
batches_per_chunk = 1
"Reload the parameters from last time and continue, or start anew when you run this config file again"
restart_from_save = False
"After how many chunks should you save parameters. Keep this number high for better performance. It will always store at end anyway"
save_every_chunks = 1


#####################
#   preprocessing   #
#####################

"Put in here the preprocessors for your data." \
"They will be run consequently on the datadict of the dataloader in the order of your list."
preprocessors = [
    PutInTheMiddle(tag="3d",output_shape=(256,512,512))
]


#####################
#     training      #
#####################
"This is the train dataloader. We will train until this one stops loading data."
"You can set the number of epochs, the datasets and if you want it multiprocessed"
training_data = PatientDataLoader(sets=TRAINING,
                                  epochs=10.0,
                                  preprocessors=preprocessors,
                                 multiprocess=False,
                                 crash_on_exception=True,
                                  )

"Schedule the reducing of the learning rate. On indexing with the number of epochs, it should return a value for the learning rate."
learning_rate_schedule = {
    0.0: 0.0001,
    9.0: 0.00001,
}
"The function to build updates."
build_updates = lasagne.updates.adam


#####################
#    validation     #
#####################
"We do our validation after every x epochs of training"
epochs_per_validation = 1.0

"Which data do we want to validate on. We will run all validation objectives on each validation data set."
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
"This is the data which will be used for testing."
test_data = PatientDataLoader(sets=[TEST, VALIDATION, TRAINING],
                              epochs=1,
                              preprocessors=preprocessors,
                              process_last_chunk=True
                              )


#####################
#     debugging     #
#####################

"Here we return a dict with the Theano objectives we are interested in. Both for the train and validation set."
"On both sets, you may request multiple objectives! Only the one called 'objective' is used to optimize on."

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

#################
# Regular model #
#################

"Here we build a model. The model returns a dict with the requested inputs for each layer:" \
"And with the outputs it generates. You may generate multiple outputs (for analysis or for some other objectives, etc)" \
"Unused outputs don't cost in performance"
def build_model():

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