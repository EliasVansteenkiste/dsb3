"""Library implementing the data augmentations.
"""
import itertools
import numpy as np
import skimage.io
import skimage.transform

from utils.custom_warnings import deprecated


tform_identity = skimage.transform.AffineTransform()
NO_AUGMENT_PARAMS = {
    "zoom_x": 1.0,  # factor
    "zoom_y": 1.0,  # factor
    "rotate": 0.0,  # degrees
    "skew_x": 0.0,  # degrees
    "skew_y": 0.0,  # degrees
    "translate_x": 0.0,  # pixels in original image
    "translate_y": 0.0, # pixels in original image
    "flip_vert": 0.0,
}

def resize_and_augment(images, output_shape=(50, 50), augment=NO_AUGMENT_PARAMS):

    def augment_2d_image(image):
        tform = build_rescale_transform(image.shape, target_shape=output_shape, keep_aspect_ratio=False)
        tform_center, tform_uncenter = build_center_uncenter_transforms(image.shape)
        augment_tform = build_augmentation_transform(**augment)
        total_tform = tform + tform_uncenter + augment_tform + tform_center
        return fast_warp(image, total_tform, output_shape=output_shape)

    result = np.array([augment_2d_image(image) for image in images])
    return result


def fast_warp(img, tf, output_shape=(50, 50), mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params # tf._matrix is deprecated
    return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)



def build_rescale_transform(image_shape, target_shape, keep_aspect_ratio=False):
    """
    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't
    100% correct, but it shouldn't matter much in practice.
    """
    rows, cols = image_shape
    trows, tcols = target_shape

    downscale_x = 1.0 * image_shape[0] / target_shape[0]
    downscale_y = 1.0 * image_shape[1] / target_shape[1]

    if keep_aspect_ratio:
        downscale_x = max(downscale_x, downscale_y)
        downscale_y = max(downscale_x, downscale_y)

    tform_ds = skimage.transform.AffineTransform(scale=(downscale_y, downscale_x))

    # centering
    shift_x = cols / (2.0 * downscale_y) - tcols / 2.0
    shift_y = rows / (2.0 * downscale_x) - trows / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_y, shift_x))
    return tform_shift_ds + tform_ds


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(zoom_x=1.0,
                                 zoom_y=1.0,
                                 skew_x=0,
                                 skew_y=0,
                                 rotate=0,
                                 shear=0,
                                 translate_x=0,
                                 translate_y=0,
                                 flip=False,
                                 flip_vert=False,
                                 **kwargs):

    #print "Not performed transformations:", kwargs.keys()

    if flip > 0.5:
        shear += 180
        rotate += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    if flip_vert > 0.5:
        shear += 180

    zoom_translate = skimage.transform.AffineTransform(translation=(translate_y, translate_x))
    zoom_augment = skimage.transform.AffineTransform(scale=(1/zoom_y, 1/zoom_x))
    tform_augment = skimage.transform.AffineTransform(rotation=np.deg2rad(rotate), shear=np.deg2rad(shear))
    skew_x = np.deg2rad(skew_x)
    skew_y = np.deg2rad(skew_y)
    tform_skew = skimage.transform.ProjectiveTransform(matrix=np.array([[np.tan(skew_x)*np.tan(skew_y) + 1, np.tan(skew_y), 0],
                                                                        [np.tan(skew_x), 1, 0],
                                                                        [0, 0, 1]]))
    return  tform_augment + zoom_augment + zoom_translate + tform_skew
