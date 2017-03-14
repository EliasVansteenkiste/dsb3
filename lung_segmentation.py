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


def segment_HU_scan_ira2(x, threshold=-350):
    mask = np.asarray(x < threshold, dtype='int8')

    for zi in xrange(mask.shape[0]):
        skimage.segmentation.clear_border(mask[zi, :, :], in_place=True)

    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    lung_label = sorted_regions[0].label
    lung_mask = np.asarray((label_image == lung_label), dtype='int8')

    # convex hull mask
    lung_mask_convex = np.zeros_like(lung_mask)
    for i in range(lung_mask.shape[2]):
        if np.any(lung_mask[:, :, i]):
            lung_mask_convex[:, :, i] = skimage.morphology.convex_hull_image(lung_mask[:, :, i])

    # old mask inside the convex hull
    mask *= lung_mask_convex
    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    print len(sorted_regions)

    remove_r = []
    for r in sorted_regions[1:]:
        print r.centroid, r.area, r.label
        if r.area > 3:
            # make an image only containing that region
            label_image_r = label_image == r.label
            # grow the mask ?? TODO: check here
            label_image_r = scipy.ndimage.binary_dilation(label_image_r,
                                                          structure=scipy.ndimage.generate_binary_structure(3, 2))
            # compute the overlap with true lungs
            overlap = label_image_r * lung_mask
            if np.sum(overlap) == 0:
                print 'REMOVE'
                remove_r.append(r)
            print '------------------------------'

    for region in remove_r:
        print region.centroid, region.area
        for coordinates in region.coords:
            lung_mask_convex[coordinates[0], coordinates[1], coordinates[2]] = 0

    return lung_mask_convex
