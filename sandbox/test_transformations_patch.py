import os
import numpy as np
import data_transforms
import pathfinder
import utils
import utils_lung
from configuration import set_configuration, config
from utils_plots import plot_slice_3d_2, plot_2d, plot_2d_4, plot_slice_3d_3

set_configuration('configs_luna_patch', 'luna_patch_local')


def test_luna_patches_3d():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    id2zyxd = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]

    # pid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.138080888843357047811238713686'
    # luna_data_paths = [pathfinder.LUNA_DATA_PATH + '/%s.mhd' % pid]
    for k, p in enumerate(luna_data_paths):
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        # img = data_transforms.hu2normHU(img)
        id = os.path.basename(p).replace('.mhd', '')
        print id

        annotations = id2zyxd[id]

        for zyxd in annotations:
            img_out, mask = config().data_prep_function_train(img,
                                                              pixel_spacing=pixel_spacing,
                                                              p_transform=config().p_transform,
                                                              p_transform_augment=config().p_transform_augment,
                                                              patch_center=zyxd,
                                                              luna_annotations=annotations,
                                                              luna_origin=origin)
            try:
                plot_slice_3d_2(img_out, mask, 0, id)
                plot_slice_3d_2(img_out, mask, 1, id)
                plot_slice_3d_2(img_out, mask, 2, id)
            except:
                pass
        print '------------------------------------------'


if __name__ == '__main__':
    test_luna_patches_3d()
