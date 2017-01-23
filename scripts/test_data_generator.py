from application.luna import LunaDataLoader
from application.preprocessors.in_the_middle import PutInTheMiddle
from interfaces.data_loader import INPUT, OUTPUT, TRAINING
#from interfaces.preprocess import AugmentInput, RescaleInput

augmentation_parameters = {
    "zoom_x":[0.8, 1.2],
    "zoom_y":"zoom_x",
    "rotate":[-10, 10],
    "shear":[0, 0],
    "skew_x":[-22.5, 22.5],
    "skew_y":[-22.5, 22.5],
    "translate_x":[-50, 50],
    "translate_y":[-50, 50],
    "change_brightness": [0, 0],
}

preprocessors = [
                 #RescaleInput(input_scale=(0,255), output_scale=(0.0, 1.0)),
                 #AugmentInput(output_shape=(160,120),**augmentation_parameters),
                 #NormalizeInput(num_samples=100),
    PutInTheMiddle(tag="luna:3d", output_shape=(512,512,512))
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
print training_data.number_of_samples

batches = training_data.generate_batch(
    chunk_size=chunk_size,
    required_input={"luna:z-slices":(chunk_size,)}, #"luna:3d":(chunk_size,512,512,512),
    required_output=dict()#{"luna:segmentation":None, "luna:sample_id":None},
)

# import matplotlib.pyplot as plt
import numpy as np
import utils.buffering

maximum = 0
i = 0
for data in batches:
    #input = data[INPUT]["luna:3d"]
    #output = data[OUTPUT]["luna:segmentation"]
    slices = data[INPUT]["luna:z-slices"][0]
    i+=1
    if slices>maximum:
        maximum=slices
    print i,slices, maximum
