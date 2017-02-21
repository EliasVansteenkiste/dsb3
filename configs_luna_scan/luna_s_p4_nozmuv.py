import data_transforms
import data_iterators
import pathfinder
import utils
import string
import numpy as np
import lasagne as nn
import configs_luna_patch.luna_p4_nozmuv as patch_config

rng = patch_config.rng
p_transform_patch = patch_config.p_transform
filter_size = p_transform_patch['patch_size'][0]
stride = filter_size / 2
extract_middle = False
pad = stride / 2

p_transform = {'patch_size': (320, 320, 320),
               'mm_patch_size': (320, 320, 320),
               'pixel_spacing': p_transform_patch['pixel_spacing']
               }

valid_pids = patch_config.valid_pids


def data_prep_function(data, luna_annotations, pixel_spacing, luna_origin,
                       p_transform=p_transform,
                       p_transform_augment=None):
    x, annotations_tf = data_transforms.transform_scan3d(data=data,
                                                         pixel_spacing=pixel_spacing,
                                                         p_transform=p_transform,
                                                         luna_annotations=luna_annotations,
                                                         p_transform_augment=None,
                                                         luna_origin=luna_origin)
    x = data_transforms.hu2normHU(x)
    y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')
    return x, y, annotations_tf


valid_data_iterator = data_iterators.PositiveLunaDataGenerator(data_path=pathfinder.LUNA_DATA_PATH,
                                                               batch_size=1,
                                                               transform_params=p_transform,
                                                               data_prep_fun=data_prep_function,
                                                               rng=rng,
                                                               patient_ids=valid_pids,
                                                               full_batch=False, random=False, infinite=False)


def build_model():
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, patch_config.__name__.split('.')[-1])
    metadata = utils.load_pkl(metadata_path)

    print 'Build model'
    model = patch_config.build_model()
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
