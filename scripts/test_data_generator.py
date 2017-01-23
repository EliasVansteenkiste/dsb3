from application.luna import LunaDataLoader
from application.preprocessors.in_the_middle import PutInTheMiddle
from application.preprocessors.lio_augmentation import LioAugment
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

]

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
    LioAugment(tags=[])
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
    print training_data.number_of_samples

    batches = training_data.generate_batch(
        chunk_size=chunk_size,
        required_input={"luna:shape":(chunk_size,3), "luna:pixelspacing":(chunk_size,3)}, #"luna:3d":(chunk_size,512,512,512),
        required_output=dict()#{"luna:segmentation":None, "luna:sample_id":None},
    )

    # import matplotlib.pyplot as plt
    import numpy as np
    import utils.buffering

    maximum_pixels = np.zeros(shape=(3,))
    maximum_mm = np.zeros(shape=(3,))
    i = 0

    np.set_printoptions(formatter={'float_kind':lambda x: "%.1f" % x})

    for data in batches:
        #input = data[INPUT]["luna:3d"]
        #output = data[OUTPUT]["luna:segmentation"]
        shape = data[INPUT]["luna:shape"]
        spacing = data[INPUT]["luna:pixelspacing"]

        in_pixel = np.array(shape)
        in_mm = in_pixel * np.array(spacing)

        i+=1
        maximum_pixels = np.maximum(maximum_pixels, in_pixel)
        maximum_mm = np.maximum(maximum_mm, in_mm)
        print i, in_pixel, in_mm, "(", maximum_pixels, maximum_mm, ")"
