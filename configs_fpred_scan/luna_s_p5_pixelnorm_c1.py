import data_transforms
import data_iterators
import pathfinder
import utils
import string
import numpy as np
import lasagne as nn

# IMPORT A CORRECT SCAN SEGMENTATION MODEL HERE
import configs_seg_scan.luna_s_p5_pixelnorm as seg_scan_config

# IMPORT A CORRECT PATCH CLASSIFICATION MODEL HERE
import configs_class_patch.luna_c1 as patch_class_config

p_transform = {'patch_size': patch_class_config.p_transform['patch_size'],
               'mm_patch_size': patch_class_config.p_transform['mm_patch_size'],
               'pixel_spacing': patch_class_config.p_transform['pixel_spacing']
               }

valid_pids = patch_class_config.valid_pids

data_prep_function = patch_class_config.data_prep_function_valid

rng = patch_class_config.rng

valid_data_iterator = data_iterators.ValidPatchPositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                                         transform_params=p_transform,
                                                                         data_prep_fun=data_prep_function,
                                                                         rng=rng,
                                                                         batch_size=1,
                                                                         patient_ids=valid_pids,
                                                                         full_batch=True,
                                                                         random=False, infinite=False)


def build_model():
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, patch_class_config.__name__.split('.')[-1])
    metadata = utils.load_pkl(metadata_path)

    print 'Build model'
    model = patch_class_config.build_model()
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
