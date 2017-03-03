from configuration import set_configuration, config
import utils_plots
import numpy as np

set_configuration('configs_seg_scan', 'luna_s_local')

data_iter = config().valid_data_iterator
for (x, y, lung_mask, id, annotations, transform_matrices) in data_iter.generate():
    id = id[0]
    annotations = annotations[0]

    predictions_scan = lung_mask * x

    for nodule_n, zyxd in enumerate(annotations):
        utils_plots.plot_slice_3d_3(input=x[0, 0], mask=lung_mask[0, 0], prediction=predictions_scan[0, 0],
                                    axis=0, pid='-'.join([str(nodule_n), str(id[0])]), idx=zyxd)
