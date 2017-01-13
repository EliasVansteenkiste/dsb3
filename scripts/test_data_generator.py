from application.data import KaggleSFDataLoader
from interfaces.data_loader import INPUT, OUTPUT
from interfaces.preprocess import AugmentInput, RescaleInput

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
                 RescaleInput(input_scale=(0,255), output_scale=(0.0, 1.0)),
                 AugmentInput(output_shape=(160,120),**augmentation_parameters),
                 #NormalizeInput(num_samples=100),
                 ]


#####################
#     training      #
#####################
training_data = KaggleSFDataLoader(sets={"training": 1.0},
                                 epochs=0.1,
                                 preprocessors=preprocessors,
                                 multiprocess=True,
                                 crash_on_exception=True
                                )

chunk_size = 16

training_data.prepare()
print training_data.number_of_samples

batch = training_data.generate_batch(
    chunk_size=chunk_size,
    required_input={"kaggle-sf:rgb":(chunk_size,3,160,120)},
    required_output={"kaggle-sf:class":None, "kaggle-sf:sample_id":None},
)

import matplotlib.pyplot as plt
import numpy as np
import utils.buffering
plt.figure()
plt.ion()

plot = None
for data in utils.buffering.buffered_gen_mp(batch):
    input = data[INPUT]["kaggle-sf:rgb"].astype('float32')
    output = data[OUTPUT]["kaggle-sf:sample_id"]

    if plot is None:
        plot = plt.imshow(np.transpose(input[0,:,:,:],[2,1,0]), interpolation='none')
        title = plt.title(str(output[0]))

    for i in xrange(input.shape[0]):
        plot.set_data(np.transpose(input[i,:,:,:],[2,1,0]))
        title.set_text(str(output[i]))
        plt.pause(0.5)
        raw_input()