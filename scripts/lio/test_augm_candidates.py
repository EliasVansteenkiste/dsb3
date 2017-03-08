import numpy as np

from application.preprocessors.augment_fpr_candidates import AugmentFPRCandidates
from application.preprocessors.dicom_to_HU import DicomToHU
from application.preprocessors.normalize_scales import DefaultNormalizer
from application.stage1 import Stage1DataLoader
from application.luna import LunaDataLoader

from interfaces.data_loader import TRAINING, VALIDATION, TEST, INPUT, OUTPUT, TRAIN


def test_loader():

    nn_input_shape = (32,)*3
    norm_patch_shape = (32,)*3

    preprocessors = [
        AugmentFPRCandidates(
            candidates_csv="candidates_V2",
            tags=["luna:3d"],
            output_shape=nn_input_shape,
            norm_patch_shape=norm_patch_shape,
            augmentation_params={
                "scale": [1, 1, 1],  # factor
                "uniform scale": 1,  # factor
                "rotation": [0, 0, 0],  # degrees
                "shear": [0, 0, 0],  # deg
                "translation": [0, 0, 0],  # mm
                "reflection": [0, 0, 0]},  # Bernoulli p
            interp_order=1),
        DefaultNormalizer(tags=["luna:3d"])
    ]

    l = LunaDataLoader(
        only_positive=True,
        multiprocess=False,
        sets=TRAINING,
        preprocessors=preprocessors)
    l.prepare()

    chunk_size = 1

    batches = l.generate_batch(
        chunk_size=chunk_size,
        required_input={"luna:3d":(chunk_size,)+nn_input_shape, "luna:pixelspacing":(chunk_size, 3)},
        required_output={"luna:target": (chunk_size,)}
    )

    for sample in batches:
        import utils.plt

        print sample[INPUT]["luna:3d"].shape, sample[OUTPUT]["luna:target"], sample[INPUT]["luna:pixelspacing"]
        utils.plt.show_animate(np.clip(sample[INPUT]["luna:3d"][0] + 0.25, 0, 1), 50, normalize=False)


if __name__ == '__main__':
    test_loader()

