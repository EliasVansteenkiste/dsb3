import numpy as np
import data_transforms
import data_iterators
import pathfinder
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import theano.tensor as T
import utils
import nn_lung

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (40, 40, 40),
               'mm_patch_size': (40, 40, 40),
               'pixel_spacing': (1., 1., 1.)
               }
p_transform_augment = {
    'translation_range_z': [-17, 17],
    'translation_range_y': [-17, 17],
    'translation_range_x': [-17, 17],
    'rotation_range_z': [-180, 180],
    'rotation_range_y': [-180, 180],
    'rotation_range_x': [-180, 180]
}

zmuv_mean, zmuv_std = None, None


# data preparation function
def data_prep_function(data, patch_center, luna_annotations, pixel_spacing, luna_origin, p_transform,
                       p_transform_augment, **kwargs):
    x, patch_annotation_tf, annotations_tf = data_transforms.transform_patch3d(data=data,
                                                                               luna_annotations=luna_annotations,
                                                                               patch_center=patch_center,
                                                                               p_transform=p_transform,
                                                                               p_transform_augment=p_transform_augment,
                                                                               pixel_spacing=pixel_spacing,
                                                                               luna_origin=luna_origin)
    x = data_transforms.hu2normHU(x)
    x = data_transforms.zmuv(x, zmuv_mean, zmuv_std)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y


data_prep_function_train = partial(data_prep_function, p_transform_augment=p_transform_augment, p_transform=p_transform)
data_prep_function_valid = partial(data_prep_function, p_transform_augment=None, p_transform=p_transform)

# data iterators
batch_size = 1
nbatches_chunk = 4
chunk_size = batch_size * nbatches_chunk

train_valid_ids = utils.load_pkl(pathfinder.LUNA_VALIDATION_SPLIT_PATH)
train_pids, valid_pids = train_valid_ids['train'], train_valid_ids['valid']

train_data_iterator = data_iterators.PatchPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                    batch_size=chunk_size,
                                                                    transform_params=p_transform,
                                                                    data_prep_fun=data_prep_function_train,
                                                                    rng=rng,
                                                                    patient_ids=train_pids,
                                                                    full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.PatchPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                    batch_size=1,
                                                                    transform_params=p_transform,
                                                                    data_prep_fun=data_prep_function_valid,
                                                                    rng=rng,
                                                                    patient_ids=valid_pids,
                                                                    full_batch=False, random=False, infinite=False)

print 'estimating ZMUV parameters'
x_big = None
for i, (x, _, _) in zip(xrange(4), train_data_iterator.generate()):
    x_big = x if x_big is None else np.concatenate((x_big, x), axis=0)
zmuv_mean = x_big.mean()
zmuv_std = x_big.std()
print 'mean:', zmuv_mean
print 'std:', zmuv_std

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 30

validate_every = int(1. * nchunks_per_epoch)
save_every = int(0.5 * nchunks_per_epoch)

learning_rate_schedule = {
    0: 1e-5,
    int(max_nchunks * 0.4): 5e-6,
    int(max_nchunks * 0.5): 3e-6,
    int(max_nchunks * 0.6): 2e-6,
    int(max_nchunks * 0.85): 1e-6,
    int(max_nchunks * 0.95): 5e-7
}

# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 b=nn.init.Constant(0.01),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

pool3d = partial(dnn.Pool3DDNNLayer,
                 pool_size=2, mode='average_exc_pad')


def build_model():
    l_in = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])
    l_target = nn.layers.InputLayer((None, 1,) + p_transform['patch_size'])

    n_filters = 64

    l1 = pool3d(l_in)
    l1 = conv3d(l1, filter_size=1, num_filters=n_filters)
    l1 = conv3d(l1, filter_size=3, num_filters=n_filters)
    l1 = conv3d(l1, filter_size=3, num_filters=n_filters)
    l1 = nn_lung.Upscale3DLayer(l1, 2)

    l3 = conv3d(l_in, filter_size=1, num_filters=n_filters)
    l3 = conv3d(l3, filter_size=5, num_filters=n_filters)

    l4 = conv3d(l_in, filter_size=1, num_filters=n_filters)
    l4 = conv3d(l4, filter_size=3, num_filters=n_filters)
    l4 = conv3d(l4, filter_size=3, num_filters=n_filters)

    lc = nn.layers.ConcatLayer([l1, l3, l4])

    l1 = conv3d(lc, filter_size=1, num_filters=n_filters)
    l2 = conv3d(l1, n_filters)
    l_out = dnn.Conv3DDNNLayer(l2, num_filters=1,
                               filter_size=1,
                               W=nn.init.Constant(0.),
                               nonlinearity=nn.nonlinearities.sigmoid)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False, epsilon=1e-12):
    predictions = T.flatten(nn.layers.get_output(model.l_out))
    targets = T.flatten(nn.layers.get_output(model.l_target))
    dice = (2. * T.sum(targets * predictions) + epsilon) / (T.sum(predictions) + T.sum(targets) + epsilon)
    return -1. * dice


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out), learning_rate)
    return updates
