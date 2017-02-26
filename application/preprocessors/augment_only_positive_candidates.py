import numpy as np
import random

from interfaces.data_loader import INPUT, OUTPUT
from augmentation_3d import sample_augmentation_parameters, Augment3D, augment_3d


class AugmentOnlyPositiveCandidates(Augment3D):
    def __init__(self, train_valid, *args, **kwargs):
        self.train_valid = train_valid
        super(AugmentOnlyPositiveCandidates, self).__init__(*args, **kwargs)

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
        input_tags_extra += [dsn+":candidates" for dsn in datasetnames]
        input_tags_extra += [dsn+":origin" for dsn in datasetnames]
        return input_tags_extra


    def process(self, sample):
        augment_p = sample_augmentation_parameters(self.augmentation_params)

        for tag in self.tags:

            pixelspacingtag = tag.split(':')[0]+":pixelspacing"
            candidatestag = tag.split(':')[0]+":candidates"
            origintag = tag.split(':')[0]+":origin"

            assert pixelspacingtag in sample[INPUT], "tag %s not found"%pixelspacingtag
            assert candidatestag in sample[INPUT], "tag %s not found"%candidatestag
            assert origintag in sample[INPUT], "tag %s not found"%origintag

            spacing = sample[INPUT][pixelspacingtag]
            candidates = sample[INPUT][candidatestag]
            origin = sample[INPUT][origintag]


            if len(candidates) == 1:
                candidate = random.choice(candidates[0])
            elif len(candidates) == 2:
                percentage_chance = 0.5 
                if random.random() < percentage_chance:
                    candidate = random.choice(candidates[1])
                else:
                    candidate = random.choice(candidates[0])
            else:
                raise Exception("candidates is empty")

            #print 'candidate', candidate

            from application.luna import LunaDataLoader
            candidateloc = LunaDataLoader.world_to_voxel_coordinates(candidate[:3],origin=origin, spacing=spacing)

            if tag in sample[INPUT]:
                volume = sample[INPUT][tag]
                
                sample[INPUT][tag] = augment_3d(
                    volume=volume,
                    pixel_spacing=spacing,
                    output_shape=self.output_shape,
                    norm_patch_shape=self.norm_patch_shape,
                    augment_p = augment_p,
                    center_to_shift= - candidateloc
                )
                # add candidate label to output tags
                
            elif tag in sample[OUTPUT]:
                #volume = sample[OUTPUT][tag]
                sample[OUTPUT][tag] = np.int32(candidate[3])
            else:
                pass
                #raise Exception("Did not find tag which I had to augment: %s"%tag)
