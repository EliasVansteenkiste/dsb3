import numpy as np

from application.preprocessors.augmentation_3d import Augment3D
from application.preprocessors.dicom_to_HU import DicomToHU
from application.preprocessors.normalize_scales import DefaultNormalizer
from application.stage1 import Stage1DataLoader

from interfaces.data_loader import TRAINING, VALIDATION, TEST, INPUT, OUTPUT, TRAIN


def test_loader():
    nn_input_shape = (256, 256, 80)
    norm_patch_shape = (340, 340, 320)  # median
    preprocessors = [
        DicomToHU(tags=["stage1:3d"]),
        Augment3D(
            tags=["stage1:3d"],
            output_shape=nn_input_shape,
            norm_patch_shape=norm_patch_shape,
            augmentation_params={
                "scale": [1, 1, 1],  # factor
                "uniform scale": 1,  # factor
                "rotation": [5, 5, 5],  # degrees
                "shear": [0, 0, 0],  # deg
                "translation": [50, 50, 50],  # mm
                "reflection": [0, 0, 0]},  # Bernoulli p
            interp_order=1),
        DefaultNormalizer(tags=["stage1:3d"])
    ]

    l = Stage1DataLoader(
        multiprocess=False,
        sets=TRAINING,
        preprocessors=preprocessors)
    l.prepare()

    chunk_size = 1

    batches = l.generate_batch(
        chunk_size=chunk_size,
        required_input={"stage1:3d":(chunk_size,)+nn_input_shape, "stage1:pixelspacing":(chunk_size, 3)},
        required_output={"cancer_prob": (chunk_size,)}
    )

    for sample in batches:
        import utils.plt

        print sample[INPUT]["stage1:3d"].shape, sample[INPUT]["stage1:pixelspacing"]
        utils.plt.show_animate(np.clip(sample[INPUT]["stage1:3d"][0] + 0.25, 0, 1), 50, normalize=False)


if __name__ == '__main__':
    test_loader()

