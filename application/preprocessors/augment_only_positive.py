import numpy as np
import random

from interfaces.data_loader import INPUT, OUTPUT
from augmentation_3d import sample_augmentation_parameters, Augment3D, augment_3d


class AugmentOnlyPositive(Augment3D):
    def __init__(self, train_valid, *args, **kwargs):
        self.train_valid = train_valid
        super(AugmentOnlyPositive, self).__init__(*args, **kwargs)

    @property
    def extra_input_tags_required(self):
        """
        We need some extra parameters to be loaded!
        :return:
        """
        datasetnames = set()
        for tag in self.tags:
            datasetnames.add(tag.split(':')[0])

        input_tags_extra = [dsn+":pixelspacing" for dsn in datasetnames]
        input_tags_extra += [dsn+":labels" for dsn in datasetnames]
        input_tags_extra += [dsn+":origin" for dsn in datasetnames]
        return input_tags_extra


    def process(self, sample):
        augment_p = sample_augmentation_parameters(self.augmentation_params)

        for tag in self.tags:

            pixelspacingtag = tag.split(':')[0]+":pixelspacing"
            labelstag = tag.split(':')[0]+":labels"
            origintag = tag.split(':')[0]+":origin"

            assert pixelspacingtag in sample[INPUT], "tag %s not found"%pixelspacingtag
            assert labelstag in sample[INPUT], "tag %s not found"%labelstag
            assert origintag in sample[INPUT], "tag %s not found"%origintag

            spacing = sample[INPUT][pixelspacingtag]
            labels = sample[INPUT][labelstag]
            origin = sample[INPUT][origintag]

            if self.train_valid == 'valid':
                lebel = labels[0]
            elif self.train_valid == 'train':
                label = random.choice(labels)
            else:
                raise

            from application.luna import LunaDataLoader
            labelloc = LunaDataLoader.world_to_voxel_coordinates(label[:3],origin=origin, spacing=spacing)

            if tag in sample[INPUT]:
                volume = sample[INPUT][tag]
                
                sample[INPUT][tag] = augment_3d(
                    volume=volume,
                    pixel_spacing=spacing,
                    output_shape=self.output_shape,
                    norm_patch_shape=self.norm_patch_shape,
                    augment_p = augment_p,
                    center_to_shift= - labelloc
                )
            elif tag in sample[OUTPUT]:
                volume = sample[OUTPUT][tag]

                sample[OUTPUT][tag] = augment_3d(
                    volume=volume,
                    pixel_spacing=spacing,
                    output_shape=self.output_shape,
                    norm_patch_shape=self.norm_patch_shape,
                    augment_p=augment_p,
                    center_to_shift= - labelloc,                     
                    cval=0.0
                )
            else:
                pass
                #raise Exception("Did not find tag which I had to augment: %s"%tag)
