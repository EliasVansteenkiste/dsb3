import os
import numpy as np
import data_transforms
import pathfinder
import utils
import utils_lung
from configuration import set_configuration, config
from utils_plots import plot_slice_3d_2, plot_2d, plot_2d_4, plot_slice_3d_3
import utils_lung
import lung_segmentation

# set_configuration('configs_seg_scan', 'luna_s_local')

# p_transform = {'patch_size': (416, 416, 416),
#                'mm_patch_size': (416, 416, 416),
#                'pixel_spacing': (1., 1., 1.)
#                }


def test_dsb3d():
    image_dir = utils.get_dir_path('analysis', pathfinder.METADATA_PATH)
    image_dir = image_dir + '/test_luna/'
    utils.auto_make_dir(image_dir)

    #id2zyxd = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

    # dsb_data_paths = ['problem_patients/154a79706bcecd0402b913d4bee9eed7/', 
    #                     'problem_patients/122c5c959fd98036c9972eec2062dc59/', 
    #                     'problem_patients/0121c2845f2b7df060945b072b2515d7/',
    #                     'problem_patients/081f4a90f24ac33c14b61b97969b7f81/',
    #                     'problem_patients/0030a160d58723ff36d73f41b170ec21/',
    #                     'problem_patients/19f3b4dea7af5d6e13acb472d6af23d8/',
    #                     'problem_patients/17ffaa3e8b53cc48e97fc6b87114e6dd/',
    #                     'problem_patients/15aa585fb2d3018b295df8619f2d1cf7/',
    #                     'problem_patients/14c534e0b7c3176d9106c6215d0aa8c6/',
    #                     'problem_patients/09b1c678fc1009d84a038cd879be4198/',
    #                     'problem_patients/0f5ab1976a1b1ef1c2eb1d340b0ce9c4/',
    #                     'problem_patients/0c98fcb55e3f36d0c2b6507f62f4c5f1/',
    #                     'problem_patients/0c9d8314f9c69840e25febabb1229fa4/']


    # dsb_data_paths = [  'problem_patients/19f3b4dea7af5d6e13acb472d6af23d8/',
    #                     'problem_patients/081f4a90f24ac33c14b61b97969b7f81/',
    #                     'problem_patients/15aa585fb2d3018b295df8619f2d1cf7/',
    #                     'problem_patients/14c534e0b7c3176d9106c6215d0aa8c6/'
    #                     ]


    # new_dsb_data_paths = [  'problem_patients/00cba091fa4ad62cc3200a657aeb957e/',
    #                     'problem_patients/01de8323fa065a8963533c4a86f2f6c1/',
    #                     'problem_patients/02801e3bbcc6966cb115a962012c35df/',
    #                     'problem_patients/07abb7bec548d1c0ccef088ce934e517/',
    #                     'problem_patients/07bca4290a2530091ce1d5f200d9d526/',
    #                     'problem_patients/0ff552aa083ecfabaf1cfd65b0a8e674/',
    #                     'problem_patients/11616de262f844e6542d3c65d9238b6e/',
    #                     'problem_patients/11fe5426ef497bc490b9f1465f1fb25/',
    #                     'problem_patients/122c5c959fd98036c9972eec2062dc59/',
    #                     'problem_patients/14afefc82d992018c485949285d20c03/',
    #                     'problem_patients/14c534e0b7c3176d9106c6215d0aa8c6/',
    #                     'problem_patients/154a79706bcecd0402b913d4bee9eed7/',
    #                     'problem_patients/15aa585fb2d3018b295df8619f2d1cf7/',
    #                     'problem_patients/174a9fc87f54d6def3730954fbafc99d/',
    #                     'problem_patients/17f5ae9fa49c4e47624f344d29bd03eb/',
    #                     'problem_patients/17ffaa3e8b53cc48e97fc6b87114e6dd/',
    #                     'problem_patients/197e035d3aed52b5a2a0de3ee4d5fcea/',
    #                     'problem_patients/199ff05d08ade6e298d37cc542bc3565/',
    #                     'problem_patients/19f3b4dea7af5d6e13acb472d6af23d8/',
    #                     'problem_patients/1b7ca8dad5c36feb0a6abf8079173e22/',
    #                     'problem_patients/1edf5480bf676f8342a7d516bab58fa0/',
    #                     'problem_patients/20c73e37fc89e007ae5b242480830860/',
    #                     'problem_patients/21bdacdcdb87bf401f34d5d582247c77/',
    #                     ]

    dsb_data_paths = [  'problem_patients/51fbac477a3639f983904fc4d42b8c15/'
                        ]

    # candidates = utils.load_pkl(
    #     'problem_patients/11616de262f844e6542d3c65d9238b6e.pkl')

    # candidates = candidates[:4]
    # print candidates
    # print '--------------'

    for k, p in enumerate(dsb_data_paths):
        pid = p.split('/')[-2]
        print pid
        img, pixel_spacing = utils_lung.read_dicom_scan(p)
        lung_mask = lung_segmentation.segment_HU_scan_elias(img, pid=pid, plot=True)



if __name__ == '__main__':
    test_dsb3d()
