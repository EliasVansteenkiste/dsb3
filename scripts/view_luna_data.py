from application.luna import LunaDataLoader
from application.preprocessors.in_the_middle import PutInTheMiddle
from application.preprocessors.augmentation_3d import Augment3D
from interfaces.data_loader import INPUT, OUTPUT, TRAINING
#from interfaces.preprocess import AugmentInput, RescaleInput
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

AUGMENTATION_PARAMETERS = {
    "scale": [1, 1, 1],  # factor
    "rotation": [360, 360, 360],  # degrees
    "shear": [0, 0, 0],  # degrees
    "translation": [128, 128, 128],  # mm
    "reflection": [0, 0, 0] #Bernoulli p
}

preprocessors = [
    Augment3D(tags=["luna:3d", "luna:segmentation"],
               output_shape=(128,128,128),
               norm_patch_shape=(128,128,128),
               augmentation_params=AUGMENTATION_PARAMETERS
               )
                 #RescaleInput(input_scale=(0,255), output_scale=(0.0, 1.0)),
                 #AugmentInput(output_shape=(160,120),**augmentation_parameters),
                 #NormalizeInput(num_samples=100),

]

#####################
#     training      #
#####################
training_data = LunaDataLoader(sets=TRAINING,
                                 epochs=1,
                                 preprocessors=preprocessors,
                                 multiprocess=False,
                                 crash_on_exception=True
                                )

chunk_size = 1
training_data.prepare()

if False:
    #####################
    #      single       #
    #####################

    sample = training_data.load_sample(0,input_keys_to_do=["luna:3d"], output_keys_to_do=["luna:segmentation"])
    data = sample[INPUT]["luna:3d"][:,:,:]
    segm = sample[OUTPUT]["luna:segmentation"][:,:,:]


else:
    batches = training_data.generate_batch(
        chunk_size=chunk_size,
        required_input={"luna:3d":(1,128,128,128)}, #"luna:3d":(chunk_size,512,512,512),
        required_output={"luna:segmentation":None,"sample_id":None},
    )
    sample = next(batches)  # first one has no tumors
    sample = next(batches)
    print "ids:", sample['ids']
    data = sample[INPUT]["luna:3d"][0,:,:,:]
    segm = sample[OUTPUT]["luna:segmentation"][0,:,:,:]

print np.min(data), np.max(data)
data = (data+500)/(900)
print data.shape, segm.shape

def get_data_step(step):
    return np.concatenate([data[:,:,step,None], data[:,:,step,None], segm[:,:,step,None]], axis=-1)

data
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
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=data.shape[2], interval=200, blit=True)

plt.show()

