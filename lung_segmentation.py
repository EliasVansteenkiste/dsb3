import numpy as np
import skimage.measure
import skimage.segmentation
import skimage.morphology
import skimage.filters
import scipy.ndimage


def segment_HU_scan(x):
    mask = np.asarray(x < -350, dtype='int32')
    for iz in xrange(mask.shape[0]):
        skimage.segmentation.clear_border(mask[iz], in_place=True)
        label_image = skimage.measure.label(mask[iz])

        areas = [r.area for r in skimage.measure.regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in skimage.measure.regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0

        selem = skimage.morphology.disk(2)
        binary = skimage.morphology.binary_erosion(binary, selem)

        selem = skimage.morphology.disk(10)
        binary = skimage.morphology.binary_closing(binary, selem)

        edges = skimage.filters.roberts(binary)
        binary = scipy.ndimage.binary_fill_holes(edges)
        mask[iz] = binary

    return mask
