from configuration import set_configuration, config
import utils_plots
import numpy as np

set_configuration('configs_seg_scan', 'luna_s_local')

data_iter = config().data_iterator
for (x_batch, tf_matrix, pid_batch) in data_iter.generate():
    print pid_batch
    id = pid_batch[0]
    utils_plots.plot_slice_3d_2(x_batch[0, 0, :, :, :], x_batch[0, 0, :, :, :], 1, id)

    # for i in xrange(x_batch.shape[0]):
    # utils_plots.plot_slice_3d_2(x_batch[i, 0], x_batch[i, 0], 0, id, idx=np.array([16, 16, 16]))
    # utils_plots.plot_slice_3d_2(x_batch[i, 0], x_batch[i, 0], 1, id)
    # utils_plots.plot_slice_3d_2(x_batch[i, 0], x_batch[i, 0], 2, id)
