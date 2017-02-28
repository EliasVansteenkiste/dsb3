import numpy as np
import random
import csv
from collections import defaultdict

from interfaces.data_loader import INPUT, OUTPUT
from augmentation_3d import sample_augmentation_parameters, Augment3D, augment_3d
from application.luna import LunaDataLoader
from utils import paths


class AugmentFPRCandidates(Augment3D):

    def __init__(self, candidates_csv, *args, **kwargs):
        super(AugmentFPRCandidates, self).__init__(*args, **kwargs)

        candidates = defaultdict(lambda: defaultdict(list))
        candidates_path = paths.LUNA_LABELS_PATH.replace("annotations", candidates_csv)

        with open(candidates_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader)  # skip the header
            for row in reader:
                c = (float(row[1]), float(row[2]), float(row[3]), int(row[4]))
                candidates[str(row[0])][int(row[4])].append(c)

        self.candidates = candidates

    @property
    def extra_input_tags_required(self):
        datasetnames = set()
        for tag in self.tags:
            datasetnames.add(tag.split(':')[0])

        input_tags_extra = [dsn+":pixelspacing" for dsn in datasetnames]
        input_tags_extra += [dsn+":origin" for dsn in datasetnames]
        input_tags_extra += [dsn+":patient_id" for dsn in datasetnames]
        input_tags_extra += [dsn+":3d" for dsn in datasetnames]
        return input_tags_extra


    def process(self, sample):
        augment_p = sample_augmentation_parameters(self.augmentation_params)

        tag = self.tags[0]
        basetag = tag.split(':')[0]

        pixelspacingtag = basetag+":pixelspacing"
        patient_idtag = basetag+":patient_id"
        origintag = basetag+":origin"

        spacing = sample[INPUT][pixelspacingtag]
        patient_id = sample[INPUT][patient_idtag]
        candidates = self.candidates[patient_id]
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

        candidateloc = LunaDataLoader.world_to_voxel_coordinates(candidate[:3],origin=origin, spacing=spacing)

        volume = sample[INPUT][basetag+":3d"]

        sample[INPUT][basetag+":3d"] = augment_3d(
            volume=volume,
            pixel_spacing=spacing,
            output_shape=self.output_shape,
            norm_patch_shape=self.norm_patch_shape,
            augment_p = augment_p,
            center_to_shift= - candidateloc
        )
        # add candidate label to output tags

        sample[OUTPUT][basetag+":target"] = np.int64(candidate[3])
