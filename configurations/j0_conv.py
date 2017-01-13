import lasagne
import theano.tensor as T
import numpy as np

from lasagne.layers import GlobalPoolLayer
from application.objectives import InterpolatedAucObjective
from application.data import SeizureDataLoader, AUC_TRAINING
from deep_learning.spectogram import SpectogramLayer
from deep_learning.varia import SelectLayer, PoolOverTimeLayer
from interfaces.data_loader import VALIDATION, TRAINING, TEST
from interfaces.preprocess import NormalizeInput
from deep_learning.deep_learning_layers import ConvolutionLayer, PoolLayer

#####################
#   running speed   #
#####################

batch_size = 64
batches_per_chunk = 1
restart_from_save = False
save_every_chunks = 1


#####################
#   preprocessing   #
#####################

preprocessors = []


#####################
#     training      #
#####################

training_data = SeizureDataLoader(sets=TRAINING,
                                  epochs=10.0,
                                  preprocessors=preprocessors
                                  )

auc_training_data =  SeizureDataLoader(sets=TRAINING,
                                  epochs=1.0,
                                  preprocessors=preprocessors,
                                  process_last_chunk=True,
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
    "validation set": SeizureDataLoader(sets=VALIDATION,
                                        epochs=1,
                                        preprocessors=preprocessors,
                                        process_last_chunk=True,
                                        ),
    "validation with auc": SeizureDataLoader(sets=[VALIDATION, AUC_TRAINING],
                                                epochs=1,
                                                preprocessors=preprocessors,
                                                process_last_chunk=True,
                                                ),
    "training set":  SeizureDataLoader(sets=TRAINING,
                                        epochs=1,
                                        preprocessors=preprocessors,
                                        process_last_chunk=True,
                                        ),
    }


#####################
#      testing      #
#####################

test_data = SeizureDataLoader(sets=[TRAINING, TEST, VALIDATION, AUC_TRAINING],
                              epochs=1,
                              preprocessors=preprocessors,
                              process_last_chunk=True
                              )


#####################
#     debugging     #
#####################
dump_network_loaded_data = False


def build_objectives(interface_layers):
    obj = InterpolatedAucObjective(interface_layers["outputs"], target_name="kaggle-seizure")
    return {
        "train":{
            "objective": obj,
        },
        "validate":{
            "objective": obj,
        },
        "test":{
            "objective": obj,
        }
    }

def build_model():

    #################
    # Regular model #
    #################

    l0 = lasagne.layers.InputLayer(shape=(None,16,120000))
    l_patient_index = lasagne.layers.InputLayer(shape=(None,))
    l_select = lasagne.layers.InputLayer(shape=(None,16,120000))

    patient_specific = []
    for i in range(3):
        l_spec = SpectogramLayer(l0,
                                 num_units=16,
                                 f = np.array(range(25,41),dtype='float32'),
                                 nonlinearity=None,
                                 window_size=41,
                                 stride=20)
        # (16, 16, 16, 5998)
        for j in xrange(3):
            l1 = ConvolutionLayer(l_spec,
                                    filter_mask_size = (3,3),
                                    filter_shape=(16, ),
                                    W=lasagne.init.Orthogonal("relu"),
                                    b=lasagne.init.Constant(0.0))

            l1 = ConvolutionLayer(l1,
                                    filter_mask_size = (3,3),
                                    filter_shape=(16, ),
                                    W=lasagne.init.Orthogonal("relu"),
                                    b=lasagne.init.Constant(0.0))
            l1 = PoolLayer(l1, pool_size=(2,4))

        l_pat = l1
        patient_specific.append(l_pat)
    l_all_patients = SelectLayer(layers=patient_specific,
                                 select_layer=l_patient_index)

    l_global1 = PoolOverTimeLayer(l_all_patients, pool_function=lambda x, *args, **kwargs: T.max(x, *args, **kwargs))
    l_global2 = PoolOverTimeLayer(l_all_patients, pool_function=lambda x, *args, **kwargs: T.mean(x, *args, **kwargs))
    l_global3 = PoolOverTimeLayer(l_all_patients, pool_function=lambda x, *args, **kwargs: T.mean(x**2, *args, **kwargs))
    l_global4 = PoolOverTimeLayer(l_all_patients, pool_function=lambda x, *args, **kwargs: T.mean(x**3, *args, **kwargs))

    l_global = lasagne.layers.ConcatLayer([l_global1, l_global2, l_global3, l_global4])

    l_dense = lasagne.layers.DenseLayer(l_global,
                                         num_units=64,
                                         W=lasagne.init.Orthogonal("relu"),
                                         b=lasagne.init.Constant(0.0))

    l_dense = lasagne.layers.DenseLayer(l_dense,
                                         num_units=1,
                                         W=lasagne.init.Orthogonal(gain=1),
                                         b=None,
                                         nonlinearity=lasagne.nonlinearities.identity)
    l_output = lasagne.layers.reshape(l_dense, shape=(-1,))

    return {
        "inputs":{
            "kaggle-seizure:default": l0,
            "kaggle-seizure:patient": l_patient_index
        },
        "outputs": {
            "predicted_rank": l_output
        },
    }