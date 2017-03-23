import lung_segmentation
import data_transforms
import data_iterators
import pathfinder
import utils
import string
import numpy as np
import lasagne as nn
import os
import utils_lung
# TODO: IMPORT A CORRECT PATCH MODEL HERE
import configs_seg_patch.luna_p8a1 as patch_config
from functools import partial
import lasagne.layers.dnn as dnn
from collections import namedtuple

# check if some predictions were generated
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/dsb_s2_p8a1_ls_elias'  # TODO write it here correctly
exclude_pids = []
if os.path.isdir(outputs_path):
    exclude_pids = os.listdir(outputs_path)
    exclude_pids = [utils_lung.extract_pid_filename(p) for p in exclude_pids]
#exclude_pids.append('b8bb02d229361a623a4dc57aa0e5c485')  # TODO hack here!

# calculate the following things correctly!
p_transform = {'patch_size': (416, 416, 416),
               'mm_patch_size': (416, 416, 416),
               'pixel_spacing': patch_config.p_transform['pixel_spacing']
               }
window_size = 144
stride = 112
n_windows = (p_transform['patch_size'][0] - window_size) / stride + 1


def data_prep_function(data, pixel_spacing, p_transform=p_transform):
    # TODO: MAKE SURE THAT DATA IS PREPROCESSED THE SAME WAY
    lung_mask = lung_segmentation.segment_HU_scan_elias(data)
    x, tf_matrix, lung_mask_out = data_transforms.transform_scan3d(data=data,
                                                                   pixel_spacing=pixel_spacing,
                                                                   p_transform=p_transform,
                                                                   lung_mask=lung_mask,
                                                                   p_transform_augment=None)
    x = data_transforms.pixelnormHU(x)
    return x, lung_mask_out, tf_matrix


print 'pathfinder.DATA_PATH', pathfinder.DATA_PATH
data_iterator = data_iterators.DSBScanLungMaskDataGeneratorFix(data_path=pathfinder.DATA_PATH,
                                                            transform_params=p_transform,
                                                            data_prep_fun=data_prep_function,
                                                            exclude_pids=exclude_pids)


# model
conv3d = partial(dnn.Conv3DDNNLayer,
                 filter_size=3,
                 pad='valid',
                 W=nn.init.Orthogonal('relu'),
                 b=nn.init.Constant(0.0),
                 nonlinearity=nn.nonlinearities.identity)

max_pool3d = partial(dnn.MaxPool3DDNNLayer,
                     pool_size=2)


def conv_prelu_layer(l_in, n_filters):
    l = conv3d(l_in, n_filters)
    l = nn.layers.ParametricRectifierLayer(l)
    return l

def build_model_patch(l_in=None, patch_size=None):
    patch_size = p_transform['patch_size'] if patch_size is None else patch_size
    l_in = nn.layers.InputLayer((None,) + patch_size) if l_in is None else l_in
    l = nn.layers.DimshuffleLayer(l_in, pattern=(0, 'x', 1, 2, 3))
    l_target = nn.layers.InputLayer((None, 1) + patch_size)

    net = {}
    base_n_filters = 128
    net['contr_1_1'] = conv_prelu_layer(l, base_n_filters)
    net['contr_1_2'] = conv_prelu_layer(net['contr_1_1'], base_n_filters)
    net['contr_1_3'] = conv_prelu_layer(net['contr_1_2'], base_n_filters)
    net['pool1'] = max_pool3d(net['contr_1_3'])

    net['encode_1'] = conv_prelu_layer(net['pool1'], base_n_filters)
    net['encode_2'] = conv_prelu_layer(net['encode_1'], base_n_filters)
    net['encode_3'] = conv_prelu_layer(net['encode_2'], base_n_filters)
    net['encode_4'] = conv_prelu_layer(net['encode_3'], base_n_filters)

    net['upscale1'] = nn.layers.Upscale3DLayer(net['encode_4'], 2)
    net['concat1'] = nn.layers.ConcatLayer([net['upscale1'], net['contr_1_3']],
                                           cropping=(None, None, "center", "center", "center"))

    net['dropout_1'] = nn.layers.dropout_channels(net['concat1'], p=0.25)

    net['expand_1_1'] = conv_prelu_layer(net['dropout_1'], 2 * base_n_filters)
    net['expand_1_2'] = conv_prelu_layer(net['expand_1_1'], base_n_filters)
    net['expand_1_3'] = conv_prelu_layer(net['expand_1_2'], base_n_filters)
    net['expand_1_4'] = conv_prelu_layer(net['expand_1_3'], base_n_filters)
    net['expand_1_5'] = conv_prelu_layer(net['expand_1_4'], base_n_filters)

    l_out = dnn.Conv3DDNNLayer(net['expand_1_5'], num_filters=1,
                               filter_size=1,
                               nonlinearity=nn.nonlinearities.sigmoid)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)

def build_model():
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, patch_config.__name__.split('.')[-1])
    metadata = utils.load_pkl(metadata_path)

    print 'Build model'
    model = build_model_patch(patch_size=(window_size, window_size, window_size))
    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])

    shp = model.l_out.output_shape
    l_out = nn.layers.ReshapeLayer(model.l_out, (-1,)+shp[-3:])
    model = namedtuple('Model', ['l_in', 'l_out', 'l_target'])(model.l_in, l_out, model.l_target)

    all_layers = nn.layers.get_all_layers(model.l_out)
    num_params = nn.layers.count_params(model.l_out)
    print '  number of parameters: %d' % num_params
    print string.ljust('  layer output shapes:', 36),
    print string.ljust('#params:', 10),
    print 'output shape:'
    for layer in all_layers:
        name = string.ljust(layer.__class__.__name__, 32)
        num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
        num_param = string.ljust(num_param.__str__(), 10)
        print '    %s %s %s' % (name, num_param, layer.output_shape)


    return model
