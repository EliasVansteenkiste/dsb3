import data_transforms
import data_iterators
import pathfinder
import utils
import string
import numpy as np
import lasagne as nn
import utils_lung
import os

# TODO: IMPORT A CORRECT SEGMENTATION PREDICTION HERE
seg_config_name = 'dsb_s5_p8a1'

# TODO: IMPORT A CORRECT PATCH CLASSIFICATION MODEL HERE
import configs_luna_props_patch.r_elias_10 as patch_class_config

p_transform = patch_class_config.p_transform

data_prep_function = patch_class_config.partial(patch_class_config.data_prep_function,
                                                p_transform_augment=None,
                                                p_transform=p_transform,
                                                world_coord_system=False,
                                                luna_origin=None)

rng = patch_class_config.rng

# candidates after segmentations path
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
segmentation_outputs_path = predictions_dir + '/%s' % seg_config_name
id2candidates_path = utils_lung.get_candidates_paths(segmentation_outputs_path)

# filter our those, who are already generated
predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
outputs_path = predictions_dir + '/dsb_tel6_s5_p8a1'  # TODO write it here correctly
exclude_pids = []
if os.path.isdir(outputs_path):
    exclude_pids = os.listdir(outputs_path)
    exclude_pids = [utils_lung.extract_pid_filename(p) for p in exclude_pids]
    print exclude_pids
    # for pid in exclude_pids:
    #     id2candidates_path.pop(pid, None)

data_iterator = data_iterators.CandidatesDSBDataGenerator(data_path=pathfinder.DATA_PATH,
                                                          transform_params=p_transform,
                                                          data_prep_fun=data_prep_function,
                                                          id2candidates_path=id2candidates_path)


def build_model():
    metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
    metadata_path = utils.find_model_metadata(metadata_dir, patch_class_config.__name__.split('.')[-1])
    print 'loading model', metadata_path
    print 'please check if model pkl is the correct one'
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
