import os
import numpy as np
import data_transforms
import pathfinder
import utils
import utils_lung
from configuration import set_configuration, config
from utils_plots import plot_slice_3d_2, plot_2d, plot_2d_4, plot_slice_3d_3, plot_2d_animation
import utils_lung
import lung_segmentation

set_configuration('configs_seg_scan', 'luna_s_local')

p_transform = {'patch_size': (416, 416, 416),
               'mm_patch_size': (416, 416, 416),
               'pixel_spacing': (1., 1., 1.)
               }


def test_luna3d():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    id2zyxd = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = [
        '/mnt/sda3/data/kaggle-lung/luna_test_patient/1.3.6.1.4.1.14519.5.2.1.6279.6001.877026508860018521147620598474.mhd']
    candidates = utils.load_pkl(
        '/mnt/sda3/data/kaggle-lung/luna_test_patient/1.3.6.1.4.1.14519.5.2.1.6279.6001.877026508860018521147620598474.pkl')

    candidates = candidates[:4]
    print candidates
    print '--------------'
    print id2zyxd['1.3.6.1.4.1.14519.5.2.1.6279.6001.877026508860018521147620598474']

    for k, p in enumerate(luna_data_paths):
        id = os.path.basename(p).replace('.mhd', '')
        print id
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        lung_mask = lung_segmentation.segment_HU_scan_elias(img)
        print np.min(lung_mask), np.max(lung_mask)
        x, annotations_tf, tf_matrix, lung_mask_out = data_transforms.transform_scan3d(data=img,
                                                                                       pixel_spacing=pixel_spacing,
                                                                                       p_transform=p_transform,
                                                                                       luna_annotations=candidates,
                                                                                       p_transform_augment=None,
                                                                                       luna_origin=origin,
                                                                                       lung_mask=lung_mask,
                                                                                       world_coord_system=False)

        print np.min(lung_mask_out), np.max(lung_mask_out)

        plot_slice_3d_2(x, lung_mask_out, 0, id)
        plot_slice_3d_2(x, lung_mask_out, 1, id)
        plot_slice_3d_2(x, lung_mask_out, 2, id)

        # for zyxd in annotations_tf:
        #     plot_slice_3d_2(x, lung_mask_out, 0, id, idx=zyxd)
        #     plot_slice_3d_2(x, lung_mask_out, 1, id, idx=zyxd)
        #     plot_slice_3d_2(x, lung_mask_out, 2, id, idx=zyxd)

        for i in xrange(136, x.shape[1]):
            plot_slice_3d_2(x, lung_mask_out, 1, str(id) + str(i), idx=np.array([200, i, 200]))

            # plot_slice_3d_2(x, lung_mask_out, 0, id, idx=np.array(x.shape) / 2)
            # plot_slice_3d_2(x, lung_mask_out, 1, id, idx=np.array(x.shape) / 2)
            # plot_slice_3d_2(x, lung_mask_out, 2, id, idx=np.array(x.shape) / 2)


def test_luna3d_2():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    id2zyxd = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

    luna_data_paths = [
        '/mnt/sda3/data/kaggle-lung/luna_test_patient/1.3.6.1.4.1.14519.5.2.1.6279.6001.943403138251347598519939390311.mhd']

    for k, p in enumerate(luna_data_paths):
        id = os.path.basename(p).replace('.mhd', '')
        print id
        img, origin, pixel_spacing = utils_lung.read_mhd(p)
        lung_mask = lung_segmentation.segment_HU_scan_ira_v2(img)
        annotations = id2zyxd[id]
        x, annotations_tf, tf_matrix, lung_mask_out = data_transforms.transform_scan3d(data=img,
                                                                                       pixel_spacing=pixel_spacing,
                                                                                       p_transform=p_transform,
                                                                                       luna_annotations=annotations,
                                                                                       p_transform_augment=None,
                                                                                       luna_origin=origin,
                                                                                       lung_mask=lung_mask,
                                                                                       world_coord_system=True)

        y = data_transforms.make_3d_mask_from_annotations(img_shape=x.shape, annotations=annotations_tf, shape='sphere')

        for zyxd in annotations_tf:
            plot_slice_3d_3(x, lung_mask_out, y, 0, id, idx=zyxd)
            plot_slice_3d_3(x, lung_mask_out, y, 1, id, idx=zyxd)
            plot_slice_3d_3(x, lung_mask_out, y, 2, id, idx=zyxd)

            # for i in xrange(136, x.shape[1]):
            #     plot_slice_3d_2(x, lung_mask_out, 1, str(id) + str(i), idx=np.array([200, i, 200]))
            #
            # plot_slice_3d_2(x, lung_mask_out, 0, id, idx=np.array(x.shape) / 2)
            # plot_slice_3d_2(x, lung_mask_out, 1, id, idx=np.array(x.shape) / 2)
            # plot_slice_3d_2(x, lung_mask_out, 2, id, idx=np.array(x.shape) / 2)


if __name__ == '__main__':
    test_luna3d()
