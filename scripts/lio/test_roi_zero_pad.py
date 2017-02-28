import numpy as np

from application.preprocessors.augment_roi_zero_pad import AugmentROIZeroPad
from application.preprocessors.dicom_to_HU import DicomToHU
from application.preprocessors.normalize_scales import DefaultNormalizer
from application.stage1 import Stage1DataLoader

from interfaces.data_loader import TRAINING, VALIDATION, TEST, INPUT, OUTPUT, TRAIN


def test_loader():

    max_rois = 10
    nn_input_shape = (32,)*3
    norm_patch_shape = (32,)*3

    preprocessors = [
        DicomToHU(tags=["stage1:3d"]),
        AugmentROIZeroPad(
            max_rois=10,
            roi_config="configurations.elias.roi_stage1_1",
            # roi_config="configurations.elias.roi_luna_4",
            tags=["stage1:3d"],
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
        required_input={"stage1:3d":(chunk_size,max_rois)+nn_input_shape, "stage1:pixelspacing":(chunk_size, 3)},
        required_output={"target": (chunk_size,)}
    )

    for sample in batches:
        import utils.plt

        print sample[INPUT]["stage1:3d"].shape, sample[INPUT]["stage1:pixelspacing"]
        for i in range(max_rois):
            utils.plt.show_animate(np.clip(sample[INPUT]["stage1:3d"][0][i] + 0.25, 0, 1), 50, normalize=False)


if __name__ == '__main__':
    test_loader()

