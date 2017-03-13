import numpy as np
import skimage.measure
import skimage.segmentation
import skimage.morphology
import skimage.filters
import scipy.ndimage
import utils_plots


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


def segment_HU_scan_elias(x, threshold=-350, pid='test'):
    mask = np.copy(x)
    binary_part = mask > threshold
    selem1 = skimage.morphology.disk(8)
    selem2 = skimage.morphology.disk(2)
    selem3 = skimage.morphology.disk(9)

    for iz in xrange(mask.shape[0]):
        # fill the body part
        filled = scipy.ndimage.binary_fill_holes(binary_part[iz])  # fill body
        filled_borders_mask = skimage.morphology.binary_erosion(filled, selem1)

        mask[iz] *= filled_borders_mask
        mask[iz] = skimage.morphology.closing(mask[iz], selem2)
        mask[iz] = skimage.morphology.erosion(mask[iz], selem3)
        mask[iz] = mask[iz] < threshold

    #discard disconnected regions
    for iz in range(mask.shape[0]/2, mask.shape[0]-1):
        overlap = mask[iz] * mask[iz+1] 
        label_image = skimage.measure.label(mask[iz+1])
        for idx, region in enumerate(skimage.measure.regionprops(label_image)):
            total_overlap = 0
            for coordinates in region.coords:                
                total_overlap += overlap[coordinates[0], coordinates[1]]
            if total_overlap < 2:
                print 'region', idx, 'in slice z=', iz-1, 'has no overlap and will be discarded'
                for coordinates in region.coords: 
                    mask[iz+1, coordinates[0], coordinates[1]] = 0


    for iz in range(mask.shape[0]/2,0,-1 ):
        overlap = mask[iz] * mask[iz-1] 
        label_image = skimage.measure.label(mask[iz-1])
        for idx, region in enumerate(skimage.measure.regionprops(label_image)):
            total_overlap = 0
            for coordinates in region.coords:                
                total_overlap += overlap[coordinates[0], coordinates[1]]
            if total_overlap < 2:
                print 'region', idx, 'in slice z=', iz-1, 'has no overlap and will be discarded'
                for coordinates in region.coords: 
                    mask[iz-1, coordinates[0], coordinates[1]] = 0

    #utils_plots.plot_all_slices(x, mask, pid, './plots/')

    return mask




if __name__ == "__main__":
    print 'main function to test lung segmentation'







