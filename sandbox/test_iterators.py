from configuration import set_configuration, config
import utils_plots

set_configuration('test_config')

data_iter = config().valid_data_iterator
for (x_batch, y_batch, pid_batch) in data_iter.generate():

    print pid_batch
    id = pid_batch[0]
    for i in xrange(x_batch.shape[0]):
        print x_batch.shape, y_batch.shape
        utils_plots.plot_slice_3d_2(x_batch[i, 0], y_batch[i, 0], 0, id)
        utils_plots.plot_slice_3d_2(x_batch[i, 0], y_batch[i, 0], 1, id)
        utils_plots.plot_slice_3d_2(x_batch[i, 0], y_batch[i, 0], 2, id)



# data_iter = config().train_data_iterator
# black, white = 0, 0
# print data_iter.nsamples
# for (x_batch, y_batch, pid_batch) in data_iter.generate():
#     id = pid_batch[0]
#     for i in xrange(x_batch.shape[0]):
#         white += np.sum(y_batch)
#         black += y_batch.size - np.sum(y_batch)
#     print white, black, black / white
