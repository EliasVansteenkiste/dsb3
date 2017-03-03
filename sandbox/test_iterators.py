from configuration import set_configuration, config
import utils_plots
import numpy as np

set_configuration('configs_seg_scan', 'luna_s_local')

data_iter = config().valid_data_iterator
for (x, y, lung_mask, annotations, transform_matrices, pid) in data_iter.generate():

    predictions_scan = lung_mask * x

    for nodule_n, zyxd in enumerate(annotations):
        utils_plots.plot_slice_3d_4(input=x[0, 0], lung_mask=lung_mask[0, 0], prediction=predictions_scan[0, 0],
                                    mask=y[0, 0],
                                    axis=0, pid='-'.join([str(nodule_n), str(pid)]), idx=zyxd)
