import os
import numpy as np
import pathfinder
import utils
import utils_lung
from configuration import set_configuration, config
from utils_plots import plot_slice_3d_2

set_configuration('configs_seg_patch', 'luna_p_local')


def test_luna_patches_3d():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    id2zyxd = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]

    for k, p in enumerate(luna_data_paths):
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        id = os.path.basename(p).replace('.mhd', '')
        print id

        annotations = id2zyxd[id]
        print annotations
        for zyxd in annotations:
            img_out, mask = config().data_prep_function_valid(img,
                                                              pixel_spacing=pixel_spacing,
                                                              p_transform=config().p_transform,
                                                              patch_center=zyxd,
                                                              luna_annotations=annotations,
                                                              luna_origin=origin)
            try:
                plot_slice_3d_2(img_out, mask, 0, id, idx=np.array(img_out.shape) / 2)
                plot_slice_3d_2(img_out, mask, 1, id, idx=np.array(img_out.shape) / 2)
                plot_slice_3d_2(img_out, mask, 2, id, idx=np.array(img_out.shape) / 2)
            except:
                pass
        print '------------------------------------------'


if __name__ == '__main__':
    test_luna_patches_3d()
