import data_transforms
import data_iterators
import pathfinder
import utils
import string
import numpy as np
import lasagne as nn
# IMPORT SCAN SEGMENTATION MODEL
import configs_seg_patch.luna_p5_pixelnorm as seg_patch_config
# IMPORT FP REDUCTION MODEL
import configs_class_patch.luna_s_p5_pixelnorm as class_patch_config


def build_patch_segmentation_model():
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, seg_patch_config.__name__.split('.')[-1])
    metadata = utils.load_pkl(metadata_path)

    print 'Build patch segmentation model'
    model = seg_patch_config.build_model()
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

    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])
    return model


def build_patch_classification_model():
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, class_patch_config.__name__.split('.')[-1])
    metadata = utils.load_pkl(metadata_path)

    print 'Build patch classification model'
    model = seg_patch_config.build_model()
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

    nn.layers.set_all_param_values(model.l_out, metadata['param_values'])
    return model
