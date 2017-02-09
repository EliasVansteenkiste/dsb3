from scipy import ndimage

from application.luna import LunaDataLoader
from application.preprocessors.in_the_middle import PutInTheMiddle
from application.preprocessors.lio_augmentation import LioAugment, AugmentOnlyPositive
from interfaces.data_loader import INPUT, OUTPUT, TRAINING
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from interfaces.preprocess import ZMUV

AUGMENTATION_PARAMETERS = {
    "scale": [1, 1, 1],  # factor
    "rotation": [0, 0, 0],  # degrees
    "shear": [0, 0, 0],  # degrees
    "translation": [5, 5, 5],  # mm
    "reflection": [0, 0, 0] #Bernoulli p
}

preprocessors = [
    # LioAugment(tags=["luna:3d", "luna:segmentation"],
    #            output_shape=(128,128,128),
    #            norm_patch_size=(128,128,128),
    #            augmentation_params=AUGMENTATION_PARAMETERS
    #            )
    # RescaleInput(input_scale=(0,255), output_scale=(0.0, 1.0)),
    #AugmentInput(output_shape=(160,120),**augmentation_parameters),
    #NormalizeInput(num_samples=100),
]

#####################
#     training      #
#####################
training_data = LunaDataLoader(
    only_positive=True,
    sets=TRAINING,
    epochs=10,
    preprocessors=preprocessors,
    multiprocess=False,
    crash_on_exception=True
)

chunk_size = 1
training_data.prepare()
data,segm = None,None
sample_nr = 0

def get_data():
    global data,segm
    global sample_nr
    while True:
        #####################
        #      single       #
        #####################
        if sample_nr>=483:
            return True
        print sample_nr
        sample = training_data.load_sample(sample_nr,input_keys_to_do=["luna:3d"], output_keys_to_do=["luna:segmentation"])
        data = sample[INPUT]["luna:3d"][:,:,:]
        segm = sample[OUTPUT]["luna:segmentation"][:,:,:]
        sample_nr += 1

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
    return False

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

bin_edges = None
total_all = None
total_tumor = None

MAX = 1500
MIN = -1500
RANGE = range(MIN,MAX,10)
while True:
    if get_data():
        break
    valid_data = data[np.logical_and(MIN<=data, data<=MAX)]
    if total_all is None:

        hist, bin_edges = np.histogram(valid_data, range=(MIN,MAX), bins = len(RANGE), density=False)
        total_all = np.array(hist)
        hist, bin_edges = np.histogram(data[segm>0.5], range=(MIN,MAX), bins = len(RANGE), density=False)
        total_tumor = np.array(hist)
    else:
        hist, bin_edges = np.histogram(valid_data, range=(MIN,MAX), bins = len(RANGE), density=False)
        total_all += np.array(hist)
        hist, bin_edges = np.histogram(data[segm>0.5], range=(MIN,MAX), bins = len(RANGE), density=False)
        total_tumor += np.array(hist)

fig = plt.figure()
ax = plt.subplot(111)
ax.bar(RANGE, -1.0*total_all/np.sum(total_all), align='edge', width=10, color=[1,0,0], linewidth=0)
ax.bar(RANGE, 1.0*total_tumor/np.sum(total_tumor), align='edge', width=10, color=[0,0,1], linewidth=0)
plt.show()

