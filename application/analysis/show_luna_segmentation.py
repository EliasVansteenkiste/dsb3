import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy import ndimage

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
    segm = kwargs['luna:gaussian']
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
    print
    print "max prediction", np.max(pred)
    print "nonzeros", np.count_nonzero(pred>0.5)
    print np.sum(pred*segm), np.sum(pred*(1-segm))
    #circle = (segm>0.5) - ndimage.binary_erosion((segm>0.5)).astype('float32')

    if true_neg==1.0:
        return

    def get_data_step(step):
        return np.concatenate([data[:,:,step,None], 0*pred[:,:,step,None]/(np.max(pred[:,:,step,None])+1e-14), segm[:,:,step,None]], axis=-1)
        return np.concatenate([data[:,:,step,None], pred[:,:,step,None]/(np.max(pred[:,:,step,None])+1e-14), circle[:,:,step,None]], axis=-1)

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


