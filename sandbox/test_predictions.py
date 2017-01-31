import os
import numpy as np
import data_transforms
import pathfinder
import utils
import utils_lung
from configuration import set_configuration, config
from utils_plots import plot_slice_3d_2, plot_2d, plot_2d_4, plot_slice_3d_3

set_configuration('test_config')


def test_luna3d():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    id2zyxd = utils_lung.read_luna_labels(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = utils_lung.get_patient_data_paths(pathfinder.LUNA_DATA_PATH)
    luna_data_paths = [p for p in luna_data_paths if '.mhd' in p]

    # luna_data_paths = [pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.223098610241551815995595311693.mhd']
    # luna_data_paths = [pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.202811684116768680758082619196.mhd']
    # luna_data_paths = [pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.174168737938619557573021395302.mhd']
    luna_data_paths = [
        pathfinder.LUNA_DATA_PATH + '/1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886.mhd']
    for k, p in enumerate(luna_data_paths):
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        id = os.path.basename(p).replace('.mhd', '')
        print id

        annotations = id2zyxd[id]

        _, annotations_out = data_transforms.transform_scan3d(img,
                                                              pixel_spacing=pixel_spacing,
                                                              p_transform=config().p_transform,
                                                              p_transform_augment=None,
                                                              # config().p_transform_augment,
                                                              luna_annotations=annotations,
                                                              luna_origin=origin)

        img_out, mask = config().data_prep_function_test(img,
                                                         pixel_spacing=pixel_spacing,
                                                         luna_annotations=annotations,
                                                         luna_origin=origin,
                                                         )

        plot_slice_3d_2(img_out, mask, 0, id)
        plot_slice_3d_2(img_out, mask, 1, id)
        plot_slice_3d_2(img_out, mask, 2, id)

        mask[mask == 0.] = 0.1
        for zyxd in annotations_out:
            plot_slice_3d_2(img_out, mask, 0, id, idx=zyxd[0])
            plot_slice_3d_2(img_out, mask, 1, id, idx=zyxd[1])
            plot_slice_3d_2(img_out, mask, 2, id, idx=zyxd[2])


if __name__ == '__main__':
    test_luna3d()
