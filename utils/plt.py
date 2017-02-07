import matplotlib.pyplot as plt
import numpy as np

def show_compare(volume1, volume2):
    plt.close('all')
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax[0, 0].imshow(volume1[volume1.shape[0] // 2], cmap="gray")
    ax[0, 1].imshow(volume1[:, volume1.shape[1] // 2], cmap="gray")
    ax[0, 2].imshow(volume1[:, :, volume1.shape[2] // 2], cmap="gray")
    ax[1, 0].imshow(volume2[volume2.shape[0] // 2], cmap="gray")
    ax[1, 1].imshow(volume2[:, volume2.shape[1] // 2], cmap="gray")
    ax[1, 2].imshow(volume2[:, :, volume2.shape[2] // 2], cmap="gray")
    plt.show()





def show_animate(data, interval=200, normalize=True):
    import matplotlib.animation as animation
    if normalize:
        mini = data.min()
        data = (data.astype("float32")-mini)/(data.max()-mini)

    def get_data_step(step):
        return np.concatenate([data[:,:,step,None], data[:,:,step,None], data[:,:,step,None]], axis=-1)

    fig = plt.figure()
    im = fig.gca().imshow(get_data_step(0))

    # initialization function: plot the background of each frame
    def init():
        im.set_data(get_data_step(0))
        return im,

    # animation function.  This is called sequentially
    def animate(i):
        im.set_data(get_data_step(i))
        return im,

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=data.shape[2], interval=interval, blit=True)
    print data.shape

    plt.show()