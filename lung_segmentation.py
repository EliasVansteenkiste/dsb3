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


def segment_HU_scan_zy(x, threshold=-350):
    mask = np.asarray(x < threshold, dtype='int8')

    for zi in xrange(mask.shape[0]):
        skimage.segmentation.clear_border(mask[zi, :, :], in_place=True)

    for yi in xrange(mask.shape[1]):
        label_image = skimage.measure.label(mask[:, yi, :])
        region_props = skimage.measure.regionprops(label_image)
        sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
        print yi
        for r in sorted_regions:
            print r.centroid, r.area, r.label
        print '-----------------------'
        if len(sorted_regions) > 2:
            for region in sorted_regions[2:]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
        mask[:, yi, :] = label_image > 0

    return mask


def segment_HU_scan_v3(x, threshold=-350):
    mask = np.asarray(x < threshold, dtype='int8')

    for zi in xrange(mask.shape[0]):
        skimage.segmentation.clear_border(mask[zi, :, :], in_place=True)

    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    for r in sorted_regions[:3]:
        print r.centroid, r.area, r.label
    print '-----------------------'
    if len(sorted_regions) > 1:
        for region in sorted_regions[1:]:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 0
    mask = label_image > 0

    return mask
