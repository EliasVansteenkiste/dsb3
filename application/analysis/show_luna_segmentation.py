import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

confusion_matrix = np.array([[]])
confusion_indices = []

def analyze(id,analysis_path,**kwargs):
    """
    Run some analysis on the output of your network (for visualization or what have you).
    Store your analysis (images, plots) in analysis_path

    The kwargs contain all the output of your network, as computed numpy tensors.
    The id is the respective id of this patient
    """
    data = kwargs['luna:3d']
    segm = kwargs['luna:segmentation']
    pred = kwargs['predicted_segmentation']

    def sigmoid(x):
      return 1. / (1. + np.exp(-x))

    data = sigmoid(data) # to get from normalized to [0,1] range

    # find accuracy table
    thresh = (pred>0.5)
    total = 1.0*np.prod(segm.shape)

    true_pos = np.count_nonzero(np.logical_and(segm,pred>0.5)) / total
    false_pos = np.count_nonzero(np.logical_and(1-segm,pred>0.5)) / total
    false_neg = np.count_nonzero(np.logical_and(segm,pred<0.5)) / total
    true_neg = np.count_nonzero(np.logical_and(1-segm,pred<0.5)) / total
    print
    print "           label=1   label=0"
    print "positive:  %.4f    %.4f"%(true_pos, false_pos)
    print "negative:  %.4f    %.4f"%(false_neg, true_neg)

    if true_neg==1.0:
        return

    def get_data_step(step):
        return np.concatenate([data[:,:,step,None], segm[:,:,step,None], pred[:,:,step,None]], axis=-1)

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

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=data.shape[2], interval=2000/data.shape[2], blit=True)
    try:
        plt.show()
    except AttributeError:
        pass


