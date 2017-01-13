import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

confusion_matrix = np.array([[]])
confusion_indices = []

def analyze(id,analysis_path,**kwargs):
    """
    kaggle-petri:default
    kaggle-petri:class
    analysis_path
    predicted_class
    """
    print kwargs.keys()
    data = kwargs["kaggle-petri:nonzero-random-section"]
    estimate = np.argmax(kwargs["predicted_class"], axis=-1)
    correct = kwargs["kaggle-petri:nonzero-random-section:class"]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,8))

    ax1.set_title("data")
    ax1.imshow(data[2,:,:], interpolation="none")
    ax2.set_title("error")
    ax2.imshow(estimate==correct, interpolation="none",vmin=0,vmax=1, cmap='RdYlGn')

    ax3.set_title("correct class")
    ax3.imshow(correct, interpolation="none",vmin=0,vmax=31)
    ax4.set_title("estimated class")
    im = ax4.imshow(estimate, interpolation="none",vmin=0,vmax=31)

    #fig.colorbar(im)

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')

    fig.tight_layout()
    fig.savefig(analysis_path+'sample_%d.pdf'%id)
    plt.close()


