import matplotlib.pyplot as plt

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