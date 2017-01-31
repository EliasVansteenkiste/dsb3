import matplotlib.pyplot as plt
import numpy as np
import pathfinder
import utils
import utils_lung
import os
import data_transforms
from configuration import set_configuration, config

set_configuration('test_config2')

# data_iter = config().test_data_iterator
# for (x_batch, y_batch, pid_batch) in data_iter.generate():
#
#     print pid_batch
#     id = pid_batch[0]
#     for i in xrange(x_batch.shape[0]):
#         print x_batch.shape, y_batch.shape
#         plot_2d_3dimg(x_batch[i, 0], y_batch[i, 0], 0, id)
#         plot_2d_3dimg(x_batch[i, 0], y_batch[i, 0], 1, id)
#         plot_2d_3dimg(x_batch[i, 0], y_batch[i, 0], 2, id)


data_iter = config().train_data_iterator
black, white = 0, 0
print data_iter.nsamples
for (x_batch, y_batch, pid_batch) in data_iter.generate():
    id = pid_batch[0]
    for i in xrange(x_batch.shape[0]):
        white += np.sum(y_batch)
        black += y_batch.size - np.sum(y_batch)
    print white, black, black / white
