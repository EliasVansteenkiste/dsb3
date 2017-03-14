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
        skimage.morphology.binary_opening(mask[iz], selem=skimage.morphology.disk(5), out=mask[iz])
        if np.sum(mask[iz]):
            mask[iz] = skimage.morphology.convex_hull_image(mask[iz])
    return mask


def segment_HU_scan_frederic(x, threshold=-350):
    mask = np.copy(x)
    binary_part = mask > threshold
    selem1 = skimage.morphology.disk(8)
    selem2 = skimage.morphology.disk(2)
    selem3 = skimage.morphology.disk(13)

    for iz in xrange(mask.shape[0]):
        # fill the body part
        filled = scipy.ndimage.binary_fill_holes(binary_part[iz])  # fill body
        filled_borders_mask = skimage.morphology.binary_erosion(filled, selem1)

        mask[iz] *= filled_borders_mask
        mask[iz] = skimage.morphology.closing(mask[iz], selem2)
        mask[iz] = skimage.morphology.erosion(mask[iz], selem3)
        mask[iz] = mask[iz] < threshold

    return mask


def segment_HU_scan_ira(x, threshold=-350):
    mask = np.asarray(x < threshold, dtype='int8')

    for zi in xrange(mask.shape[0]):
        skimage.segmentation.clear_border(mask[zi, :, :], in_place=True)

    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    lung_label = sorted_regions[0].label
    label_image = (label_image == lung_label)
    mask = np.asarray(label_image, dtype='int8')

    for i in range(mask.shape[2]):
        if np.any(mask[:, :, i]):
            mask[:, :, i] = skimage.morphology.convex_hull_image(mask[:, :, i])

    return mask
