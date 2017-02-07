import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import lasagne
import theano.tensor as T
import utils
import nn_lung

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (64, 64, 64),
               'mm_patch_size': (64, 64, 64),
               'pixel_spacing': (1., 0.7, 0.7)
               }
p_transform_augment = {
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}


# data preparation function
def data_prep_function(data, patch_center, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, mask_shape, **kwargs):
    x = data_transforms.hu2normHU(data)
    x, patch_annotation_tf = data_transforms.transform_patch3d(data=x,
                                                               luna_annotations=None,
                                                               patch_center=patch_center,
                                                               p_transform=p_transform,
                                                               p_transform_augment=p_transform_augment,
                                                               pixel_spacing=pixel_spacing,
                                                               luna_origin=luna_origin)
    print 'x.shape', x.shape
    print 'patch_annotation_tf', patch_annotation_tf

    return x


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment,
                                   p_transform=p_transform,
                                   mask_shape='sphere')
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None,
                                   p_transform=p_transform,
                                   mask_shape='sphere')

# data iterators
batch_size = 1
nbatches_chunk = 4
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.Luna_DG_Elias(data_path=pathfinder.LUNA_DATA_PATH,
                                                                    batch_size=chunk_size,
                                                                    transform_params=p_transform,
                                                                    data_prep_fun=data_prep_function_train,
                                                                    rng=rng,
                                                                    patient_ids=train_valid_ids['train'],
                                                                    full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.Luna_DG_Elias(data_path=pathfinder.LUNA_DATA_PATH,
                                                                    batch_size=1,
                                                                    transform_params=p_transform,
                                                                    data_prep_fun=data_prep_function_valid,
                                                                    rng=rng,
                                                                    patient_ids=train_valid_ids['valid'],
                                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 30

validate_every = int(1. * nchunks_per_epoch)
save_every = int(0.5 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-5,
    int(max_nchunks * 0.5): 5e-6,
    int(max_nchunks * 0.6): 4e-6,
    int(max_nchunks * 0.7): 3e-6,
    int(max_nchunks * 0.8): 2e-6,
    int(max_nchunks * 0.9): 1e-7
}

# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)

drop = lasagne.layers.DropoutLayer

bn = lasagne.layers.batch_norm

dense = partial(lasagne.layers.DenseLayer,
    W=lasagne.init.Orthogonal('relu'),
    b=lasagne.init.Constant(0.0),
    nonlinearity=lasagne.nonlinearities.rectify)

def build_model():
    l_in = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((None, 1))

    print 'l_target.output_shape', l_target.output_shape

    print l_in.output_shape

    # n = 16
    # # l = bn(conv3d(l, n, filter_size=7, stride=2))
    # # l = max_pool3d(l)
    # l = bn(conv3d(l_in, n))
    # l = bn(conv3d(l, n))
    # l = max_pool3d(l)

    # n *= 2
    # l = bn(conv3d(l, n))
    # l = bn(conv3d(l, n))
    # l = max_pool3d(l)

    # n *= 2
    # l = bn(conv3d(l, n))
    # l = bn(conv3d(l, n))
    # l = max_pool3d(l)

    # n *= 2
    # l = bn(conv3d(l, n))
    # l = bn(conv3d(l, n))
    # l = max_pool3d(l)

    # n *= 2
    # l = bn(conv3d(l, n))
    # l = bn(conv3d(l, n))
    # l = max_pool3d(l)

    # n *= 2
    # l = bn(dense(drop(l), n))
    # l = bn(dense(drop(l), n))

    l_out = lasagne.layers.DenseLayer(l_in,
                                 num_units=2,
                                 W=lasagne.init.Constant(0.0),
                                 b=None,
                                 nonlinearity=lasagne.nonlinearities.softmax)


    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)



def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = nn.layers.get_output(model.l_out)
    predictions = T.clip(predictions, epsilon, 1.-epsilon)
    predictions = T.log(predictions)
    target = nn.layers.get_output(model.l_target)
    target = target.flatten()
    target = T.cast(target,'int32')
    predictions = predictions[T.arange(target.shape[0]), target]
    return T.mean(predictions)



def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates


