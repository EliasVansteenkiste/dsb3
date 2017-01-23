from application.luna import LunaDataLoader
from application.preprocessors.in_the_middle import PutInTheMiddle
from application.preprocessors.lio_augmentation import LioAugment
from interfaces.data_loader import INPUT, OUTPUT, TRAINING
#from interfaces.preprocess import AugmentInput, RescaleInput

"Put in here the preprocessors for your data." \
"They will be run consequently on the datadict of the dataloader in the order of your list."
preprocessors = [
    LioAugment(tags=["luna:segmentation"],
               output_shape=(256,256,256),
               norm_patch_size=(256,256,256)
               ),
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

if True:
    print training_data.number_of_samples

    batches = training_data.generate_batch(
        chunk_size=chunk_size,
        required_input={},
        required_output={"luna:segmentation":None},
    )

    # import matplotlib.pyplot as plt
    import numpy as np
    import utils.buffering

    i = 0

    np.set_printoptions(formatter={'float_kind':lambda x: "%.1f" % x})

    positive_pixels = 0
    zero_pixels = 0
    for data in batches:
        i+=1
        output = data[OUTPUT]["luna:segmentation"]
        positive_pixels += np.sum(output==1)
        zero_pixels += np.sum(output!=1)

        print i,positive_pixels,zero_pixels
