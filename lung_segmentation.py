import numpy as np
import skimage.measure
import skimage.segmentation
import skimage.morphology
import skimage.filters


def segment_HU_scan(x):
    mask = np.asarray(x < -350, dtype='int32')
    for iz in xrange(mask.shape[0]):
        skimage.segmentation.clear_border(mask[iz], in_place=True)
        skimage.morphology.binary_opening(mask[iz], selem=skimage.morphology.disk(5), out=mask[iz])
        if np.sum(mask[iz]):
            mask[iz] = skimage.morphology.convex_hull_image(mask[iz])
    return mask
