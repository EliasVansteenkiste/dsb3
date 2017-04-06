import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn

from collections import OrderedDict, namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import lasagne
import theano.tensor as T
import utils

restart_from_save = False
rng = np.random.RandomState(33)

# transformations
p_transform = {'patch_size': (48, 48, 48),
               'mm_patch_size': (48, 48, 48),
               'pixel_spacing': (1., 1., 1.)
               }

p_transform_augment = {
    'translation_range_z': [-3, 3],
    'translation_range_y': [-3, 3],
    'translation_range_x': [-3, 3],
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}

positive_proportion = 0.5

properties = ['diameter', 'calcification', 'lobulation', 'malignancy', 'margin', 'sphericity',
              'spiculation', 'texture']
nproperties = len(properties)


def label_prep_function(annotation,properties_included):
    patch_zyxd = annotation[:4]
    if patch_zyxd[-1] == 0:
        if len(properties_included)>0:
            return np.asarray([0] * len(properties_included), dtype='float32')
        else:
            return np.asarray([0] * len(properties), dtype='float32')
    else:
        label = []
        properties_dict = annotation[-1]
        if len(properties_included)>0:
            for p in properties_included:
                label.append(properties_dict[p]/5.0)
        else:
            for p in properties:
                    label.append(properties_dict[p])
    return label



# data preparation function
def data_prep_function(data, patch_center, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, world_coord_system, **kwargs):
    x, patch_annotation_tf = data_transforms.transform_patch3d(data=data,
                                                               luna_annotations=None,
                                                               patch_center=patch_center,
                                                               p_transform=p_transform,
                                                               p_transform_augment=p_transform_augment,
                                                               pixel_spacing=pixel_spacing,
                                                               luna_origin=luna_origin,
                                                               world_coord_system=world_coord_system)
    x = data_transforms.hu2normHU(x)

    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform, world_coord_system=True)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform, world_coord_system=True)

# data iterators
batch_size = 16
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']


train_data_iterator = data_iterators.CandidatesPropertiesLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                           batch_size=chunk_size,
                                                                           transform_params=p_transform,
                                                                           label_prep_fun=label_prep_function,
                                                                           nproperties=nproperties,
                                                                           data_prep_fun=data_prep_function_train,
                                                                           rng=rng,
                                                                           patient_ids=train_pids,
                                                                           full_batch=True, random=True, infinite=True,
                                                                           positive_proportion=positive_proportion,
                                                                           random_negative_samples=True,
                                                                           properties_included=["malignancy"])

valid_data_iterator = data_iterators.CandidatesLunaValidDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                      transform_params=p_transform,
                                                                      data_prep_fun=data_prep_function_valid,
                                                                      patient_ids=valid_pids,
                                                                      label_prep_fun=label_prep_function,
                                                                      properties_included=["malignancy"])

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 100

validate_every = int(1 * nchunks_per_epoch)
save_every = int(1. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-4,
    int(max_nchunks * 0.4): 6e-5,
    int(max_nchunks * 0.6): 3e-5,
    int(max_nchunks * 0.8): 1e-5,
    int(max_nchunks * 0.9): 0.5e-5
}

# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool3d = dnn.MaxPool3DDNNLayer

drop = lasagne.layers.DropoutLayer

bn = lasagne.layers.batch_norm

dense = partial(lasagne.layers.DenseLayer,
                W=lasagne.init.Orthogonal('relu'),
                b=lasagne.init.Constant(0.0),
                nonlinearity=lasagne.nonlinearities.rectify)



# we reduce # filters by factor of 8 compared to original inception-v4



conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.HeNormal('relu'),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.rectify)

n_filt_red_f = 8
def inception_resnet_v2_stem(lin):
    # in original inception-resnet-v2, conv stride is 2
    l = conv3d(lin, 32//n_filt_red_f, pad='valid')
    l = conv3d(l, 32//n_filt_red_f, pad='valid')
    l = conv3d(l, 64//n_filt_red_f, pad='same')
    # in original inception-resnet-v2, stride is 2
    a = max_pool3d(l, pool_size=3, stride=1)
    # in original inception-resnet-v2, conv stride is 2
    b = conv3d(l, 96//n_filt_red_f, pad='valid')
    l = lasagne.layers.ConcatLayer([a, b])

    a = conv3d(l, 64//n_filt_red_f, filter_size=1, pad='same')
    a = conv3d(a, 96//n_filt_red_f, filter_size=3, pad='valid')

    b = conv3d(l, 64//n_filt_red_f, filter_size=1, pad='same')
    b = conv3d(b, 64//n_filt_red_f, filter_size=(7,1,1), pad='same')
    b = conv3d(b, 64//n_filt_red_f, filter_size=(1,7,1), pad='same')
    b = conv3d(b, 64//n_filt_red_f, filter_size=(1,1,7), pad='same')
    b = conv3d(b, 96//n_filt_red_f, filter_size=3, pad='valid')
    l = lasagne.layers.ConcatLayer([a, b])


    a = conv3d(l, 192//n_filt_red_f, filter_size=3, pad='valid') # in original inception-resnet-v2, conv stride should be 2
    b = max_pool3d(l, pool_size=3, stride=1) # in original inception-resnet-v2, stride is 2
    print a.output_shape
    print b.output_shape
    l = lasagne.layers.ConcatLayer([a, b])

    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    
    return l 

def inception_resnet_v2_A(lin):
    
    a = conv3d(lin, 32//n_filt_red_f, filter_size=1, pad='same')
    
    b = conv3d(lin, 32//n_filt_red_f, filter_size=1, pad='same')
    b = conv3d(b, 32//n_filt_red_f, filter_size=3, pad='same')
    
    c = conv3d(lin, 32//n_filt_red_f, filter_size=1, pad='same')
    c = conv3d(c, 48//n_filt_red_f, filter_size=3, pad='same')
    c = conv3d(c, 64//n_filt_red_f, filter_size=3, pad='same')
    
    l = lasagne.layers.ConcatLayer([a, b, c])

    l = conv3d(l, 384//n_filt_red_f, filter_size=1, pad='same', nonlinearity=lasagne.nonlinearities.linear)

    l = lasagne.layers.ElemwiseSumLayer([l, lin])
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    
    return l


def inception_resnet_v2_reduction_A(lin):
    a = max_pool3d(lin, pool_size=3, stride=2)

    b = conv3d(lin, 384//n_filt_red_f, filter_size=3, stride=2, pad='valid')

    c = conv3d(lin, 256//n_filt_red_f, filter_size=1, pad='same')
    c = conv3d(c, 256//n_filt_red_f, filter_size=3, pad='same')
    c = conv3d(c, 384//n_filt_red_f, filter_size=3, stride=2, pad='valid')
    
    l = lasagne.layers.ConcatLayer([a, b, c])
    
    return l
    

def inception_resnet_v2_B(lin):
    a = conv3d(lin, 192//n_filt_red_f, filter_size=1, pad='same')

    b = conv3d(lin, 128//n_filt_red_f, filter_size=1, pad='same')
    b = conv3d(b, 160//n_filt_red_f, filter_size=(7,1,1), pad='same')
    b = conv3d(b, 160//n_filt_red_f, filter_size=(1,7,1), pad='same')
    b = conv3d(b, 192//n_filt_red_f, filter_size=(1,1,7), pad='same')

    l = lasagne.layers.ConcatLayer([a, b])
    l = conv3d(l, 1154//n_filt_red_f, filter_size=1, pad='same', nonlinearity=lasagne.nonlinearities.linear)

    l = lasagne.layers.ElemwiseSumLayer([l, lin])
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
   
    return l


def inception_resnet_v2_reduction_B(lin):
    a = max_pool3d(lin, pool_size=3, stride=2)

    b = conv3d(lin, 256//n_filt_red_f, filter_size=1, stride=1, pad='same')
    b = conv3d(b, 288//n_filt_red_f, filter_size=3, stride=2, pad='valid')

    c = conv3d(lin, 256//n_filt_red_f, filter_size=1, stride=1, pad='same')
    c = conv3d(c, 288//n_filt_red_f, filter_size=3, stride=2, pad='valid')


    d = conv3d(lin, 256//n_filt_red_f, filter_size=1, pad='same')
    d = conv3d(d, 288//n_filt_red_f, filter_size=3, pad='same')
    d = conv3d(d, 320//n_filt_red_f, filter_size=3, stride=2, pad='valid')
    
    l = lasagne.layers.ConcatLayer([a, b, c, d])
    
    return l


def inception_resnet_v2_C(lin):
    a = conv3d(lin, 192//n_filt_red_f, filter_size=1, pad='same')

    b = conv3d(lin, 192//n_filt_red_f, filter_size=1, pad='same')
    b = conv3d(b, 224//n_filt_red_f, filter_size=(3,1,1), pad='same')
    b = conv3d(b, 224//n_filt_red_f, filter_size=(1,3,1), pad='same')
    b = conv3d(b, 256//n_filt_red_f, filter_size=(1,1,3), pad='same')

    l = lasagne.layers.ConcatLayer([a, b])
    l = conv3d(l, 2048//n_filt_red_f, filter_size=1, pad='same', nonlinearity=lasagne.nonlinearities.linear)

    l = lasagne.layers.ElemwiseSumLayer([l, lin])
    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)
    
    return l




def build_model():
    l_in = nn.layers.InputLayer((None,) + p_transform['patch_size'])
    l_din = lasagne.layers.DimshuffleLayer(l_in, pattern=[0,'x',1,2,3])
    l_target = nn.layers.InputLayer((None,))

    num_A_blocks = 1
    num_B_blocks = 1
    num_C_blocks = 1

    l = inception_resnet_v2_stem(l_din)
    for i in range(num_A_blocks):
        l = inception_resnet_v2_A(l)
    l = inception_resnet_v2_reduction_A(l)
    for i in range(num_B_blocks):
        l = inception_resnet_v2_B(l)
    l = inception_resnet_v2_reduction_B(l)
    for i in range(num_C_blocks):
        l = inception_resnet_v2_C(l)

    l = dense(drop(l), 512)

    l_out = nn.layers.DenseLayer(l,1,nonlinearity=nn.nonlinearities.sigmoid, W=lasagne.init.Orthogonal(),
                b=lasagne.init.Constant(0))

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


d_objectives_deterministic = {}
d_objectives = {}


def build_objective(model, deterministic=False):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    objective = lasagne.objectives.squared_error(predictions,targets)
    loss = T.mean(objective)
    return loss


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
