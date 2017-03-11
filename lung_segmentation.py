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

        # areas = [r.area for r in skimage.measure.regionprops(label_image)]
        # areas.sort()
        # if len(areas) > 2:
        #     for region in skimage.measure.regionprops(label_image):
        #         if region.area < areas[-2]:
        #             for coordinates in region.coords:
        #                 label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0

        selem = skimage.morphology.disk(2)
        binary = skimage.morphology.binary_erosion(binary, selem)

        selem = skimage.morphology.disk(10)
        binary = skimage.morphology.binary_closing(binary, selem)

        edges = skimage.filters.roberts(binary)
        binary = scipy.ndimage.binary_fill_holes(edges)
        mask[iz] = binary

    return mask, None


def segment_HU_scan_fred(x):

    # resize from [-1,1] to [0,1] can be removed if thresholds are changed accordingly?
    mask = (x + 1)/2

    for iz in xrange(mask.shape[0]):

        # detect if additional borders were added.
        corner = 0
        for i in xrange(0,mask[iz].shape[1]):
           if mask[iz, i, i] == 0:
               corner = i
               break

        # select inner part
        part = mask[iz,corner:-corner-1,corner:-corner-1]
        binary_part = part > 0.5

        # fill the body part
        filled = scipy.ndimage.binary_fill_holes(binary_part) # fill body
        selem = skimage.morphology.disk(5) # clear details outside of major body part
        filled_borders = skimage.morphology.erosion(filled, selem)
        filled_borders = 1 - filled_borders # flip mask

        # put mask back
        full_mask = np.ones((mask.shape[1],mask.shape[2]))
        full_mask[corner:-corner-1,corner:-corner-1] = filled_borders

        full_mask = np.asarray(full_mask,dtype=np.bool)

        # set outside to grey
        filled_borders = mask[iz]
        filled_borders[full_mask]=0.75


        # finally do the normal segmentation operations

        # change the disk value of this operation to make it less aggressive
        selem = skimage.morphology.disk(13)
        eroded = skimage.morphology.erosion(filled_borders, selem)

        selem = skimage.morphology.disk(2)
        closed = skimage.morphology.closing(eroded, selem)



        # threshold grey values
        t = 0.25
        mask[iz] = closed < t


    return mask

def segment_HU_scan_greyvalued_original_image(x):


    mask = np.copy(x)

    for iz in xrange(mask.shape[0]):

        binary_part = mask[iz] > -350

        # fill the body part
        filled = scipy.ndimage.binary_fill_holes(binary_part) # fill body
        selem = skimage.morphology.disk(8) # clear details outside of major body part
        filled_borders_mask = skimage.morphology.binary_erosion(filled, selem)
        filled_borders_mask = np.asarray(1 - filled_borders_mask,dtype=np.bool) # flip mask


        # set outside to grey
        filled_borders = mask[iz]
        filled_borders[filled_borders_mask]=0

        # finally do the normal segmentation operations

        # change the disk value of this operation to make it less aggressive

        # reduce the noise first
        selem = skimage.morphology.disk(2)
        closed = skimage.morphology.closing(filled_borders, selem)
        # aggressive erosion
        selem = skimage.morphology.disk(13)
        eroded = skimage.morphology.erosion(closed, selem)


        # # threshold grey values
        t = -350
        mask[iz] = eroded < t

    return mask