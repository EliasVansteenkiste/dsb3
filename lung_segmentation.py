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


def segment_HU_scan_ira(x, threshold=-350, min_area=1000000):
    mask = np.asarray(x < threshold, dtype='int8')

    mask = skimage.morphology.binary_opening(mask, skimage.morphology.cube(4))
    mask = np.asarray(mask, dtype='int8')

    for zi in xrange(mask.shape[0]):
        skimage.segmentation.clear_border(mask[zi, :, :], in_place=True)

    label_image = skimage.measure.label(mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    lung_region = sorted_regions[0]

    candidate_lung_regions = []
    for r in sorted_regions:
        if r.area > min_area:
            candidate_lung_regions.append(r)
    if len(candidate_lung_regions) > 1:
        print 'NUMBER OF CANDIDATE REGIONS', len(candidate_lung_regions)
        middle_patch = label_image[mask.shape[0] / 2]
        region2distance, region2centroid = {}, {}
        for r in candidate_lung_regions:
            middle_patch_r = middle_patch == r.label
            centroid = np.average(np.where(middle_patch_r), axis=1)
            region2centroid[r] = centroid
            distance = np.sum((centroid - np.asarray(middle_patch.shape) / 2) ** 2)
            region2distance[r] = distance
        lung_region = min(region2distance, key=region2distance.get)
        n_lung_regions = 1
        for r in candidate_lung_regions:
            print region2centroid[r]
            if abs(region2centroid[r][0] - region2centroid[lung_region][0]) < 100:
                label_image[label_image == r.label] = lung_region.label
                n_lung_regions += 1

    lung_label = lung_region.label
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

    for r in sorted_regions[n_lung_regions:]:
        if r.area > 125:
            label_image_r = label_image == r.label
            for i in range(label_image_r.shape[0]):
                if np.any(label_image_r[i]):
                    label_image_r[i] = skimage.morphology.convex_hull_image(label_image_r[i])
            lung_mask_convex *= 1 - label_image_r

    return lung_mask_convex


def segment_HU_scan_elias(x, threshold=-350):
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
        mask[iz] = mask[iz] < threshold

    # params
    overlap_treshold = 7
    ratio_overlap_treshold = 0.015

    # discard disconnected regions, start at the middle slice and go to the head
    for iz in range(mask.shape[0] / 2, mask.shape[0] - 1):
        overlap = mask[iz] * mask[iz + 1]
        label_image = skimage.measure.label(mask[iz + 1])

        for idx, region in enumerate(skimage.measure.regionprops(label_image)):
            total_overlap = 0
            for coordinates in region.coords:
                total_overlap += overlap[coordinates[0], coordinates[1]]
            ratio_overlap = 1. * total_overlap / region.area

            if total_overlap < overlap_treshold or ratio_overlap < ratio_overlap_treshold:

                for coordinates in region.coords:
                    mask[iz + 1, coordinates[0], coordinates[1]] = 0

    # discard disconnected regions, start at the middle slice and go to the head
    for iz in range(mask.shape[0] / 2, 0, -1):
        overlap = mask[iz] * mask[iz - 1]
        label_image = skimage.measure.label(mask[iz - 1])
        for idx, region in enumerate(skimage.measure.regionprops(label_image)):
            total_overlap = 0
            for coordinates in region.coords:
                total_overlap += overlap[coordinates[0], coordinates[1]]
            ratio_overlap = 1. * total_overlap / region.area
            if total_overlap < overlap_treshold or ratio_overlap < ratio_overlap_treshold:
                for coordinates in region.coords:
                    mask[iz - 1, coordinates[0], coordinates[1]] = 0

    # erode out the blood vessels and the borders of the lung for a cleaner mask
    for iz in xrange(mask.shape[0]):
        mask[iz] = skimage.morphology.binary_dilation(mask[iz], selem3)

    return mask
