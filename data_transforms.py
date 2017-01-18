from collections import namedtuple
import numpy as np
import skimage.transform
import skimage.draw
from configuration import config
import skimage.exposure, skimage.filters

MAX_HU = 400.
MIN_HU = -1000.


def ct2normHU(x, metadata):
    # TODO: maybe no copy
    x_hu = np.copy(x)
    x_hu[x_hu < 0.] = 0.
    x_hu = metadata['RescaleSlope'] * x_hu + metadata['RescaleIntercept']
    x_hu = (x_hu - MIN_HU) / (MAX_HU - MIN_HU)
    x_hu[x_hu < 0.] = 0.
    x_hu[x_hu > 1.] = 1.
    return x_hu


def hu2normHU(x):
    # TODO: maybe no copy
    x_norm = np.copy(x)
    x_norm = (x_norm - MIN_HU) / (MAX_HU - MIN_HU)
    x_norm[x_norm < 0.] = 0.
    x_norm[x_norm > 1.] = 1.
    return x_norm


def sample_augmentation_parameters(transformation):
    shift_x = config().rng.uniform(*transformation.get('translation_range_x', [0., 0.]))
    shift_y = config().rng.uniform(*transformation.get('translation_range_y', [0., 0.]))
    translation = (shift_x, shift_y)
    rotation = config().rng.uniform(*transformation.get('rotation_range', [0., 0.]))
    shear = config().rng.uniform(*transformation.get('shear_range', [0., 0.]))
    roi_scale = config().rng.uniform(*transformation.get('roi_scale_range', [1., 1.]))
    z = config().rng.uniform(*transformation.get('zoom_range', [1., 1.]))
    zoom = (z, z)

    if 'do_flip' in transformation:
        if type(transformation['do_flip']) == tuple:
            flip_x = config().rng.randint(2) > 0 if transformation['do_flip'][0] else False
            flip_y = config().rng.randint(2) > 0 if transformation['do_flip'][1] else False
        else:
            flip_x = config().rng.randint(2) > 0 if transformation['do_flip'] else False
            flip_y = False
    else:
        flip_x, flip_y = False, False

    return namedtuple('Params', ['translation', 'rotation', 'shear', 'zoom',
                                 'roi_scale',
                                 'flip_x', 'flip_y'])(translation, rotation, shear, zoom,
                                                      roi_scale,
                                                      flip_x, flip_y)


def transform_3d_rescale(data, pixel_spacing, transformation,
                         mm_patch_size=(512, 512, 512),
                         mm_center_location=(0.5, 0.5, 0.5),
                         out_pixel_spacing=(1., 1., 1.)):
    """
    TODO: BIG BUGS in ZY transformation
    :param data:
    :param pixel_spacing:
    :param transformation:
    :param mm_patch_size:
    :param mm_center_location:
    :param out_pixel_spacing:
    :return:
    """
    patch_size = transformation['patch_size']
    mm_patch_size = transformation.get('mm_patch_size', mm_patch_size)

    # XY
    print 'XY'
    current_shape_yx = data.shape[1:]
    pixel_rescaling_yx = (out_pixel_spacing[1] / pixel_spacing[1], out_pixel_spacing[2] / pixel_spacing[2])
    mm_shape_yx = tuple(int(float(d) * ps) for d, ps in zip(current_shape_yx, pixel_rescaling_yx))
    mm_patch_size_yx = (mm_patch_size[1], mm_patch_size[2])
    patch_size_yx = (patch_size[1], patch_size[2])
    mm_center_location_yx = (mm_center_location[1], mm_center_location[2])

    total_tform_yx = build_2d_rescale_transform(pixel_rescaling_yx, current_shape_yx, mm_shape_yx, mm_patch_size_yx,
                                                patch_size_yx,
                                                mm_center_location_yx)

    out_data_yx = np.zeros((data.shape[0],) + patch_size_yx, dtype='float32')
    for i in xrange(data.shape[0]):
        out_data_yx[i] = fast_warp(data[i], total_tform_yx, output_shape=patch_size_yx)

    print 'ZY'
    current_shape_zy = (out_data_yx.shape[0], out_data_yx.shape[1])
    pixel_rescaling_zy = (out_pixel_spacing[0] / pixel_spacing[0], 1.)
    mm_shape_zy = (pixel_rescaling_zy[0] * current_shape_zy[0], current_shape_zy[1])
    mm_patch_size_zy = (mm_patch_size[0], current_shape_zy[1])
    patch_size_zy = (patch_size[0], current_shape_zy[1])
    mm_center_location_zy = (mm_center_location[0], mm_center_location[1])

    total_tform_zy = build_2d_rescale_transform(pixel_rescaling_zy, current_shape_zy, mm_shape_zy, mm_patch_size_zy,
                                                patch_size_zy,
                                                mm_center_location_zy)

    out_data_zy = np.zeros(patch_size_zy + (out_data_yx.shape[-1],), dtype='float32')
    for i in xrange(out_data_zy.shape[-1]):
        out_data_zy[:, :, i] = fast_warp(out_data_yx[:, :, i], total_tform_zy, output_shape=patch_size_zy)

    return out_data_yx


def build_2d_rescale_transform(downscale_factor, current_shape, mm_shape, mm_patch_size, patch_size,
                               mm_center_location):
    tform_normscale_xy = build_rescale_transform(scaling_factor=downscale_factor,
                                                 image_shape=current_shape, target_shape=mm_shape)

    tform_shift_center, tform_shift_uncenter = build_shift_center_transform(image_shape=mm_shape,
                                                                            center_location=mm_center_location,
                                                                            patch_size=mm_patch_size)

    patch_scale = (1. * mm_patch_size[0] / patch_size[0], 1. * mm_patch_size[0] / patch_size[0])
    tform_patch_scale_xy = build_rescale_transform(patch_scale, mm_patch_size, target_shape=patch_size)
    total_tform_xy = tform_patch_scale_xy + tform_shift_uncenter + tform_shift_center + tform_normscale_xy
    return total_tform_xy


def luna_transform_rescale_slice(data, annotations, pixel_spacing, p_transform, p_transform_augment,
                                 p_augment_sample=None,
                                 mm_center_location=(.5, .5)):
    """

    :param data: one slice (y,x)
    :param annptations:  dict  {'centers':[(x,y)], 'radii':[(r_x,r_y)]}
    :param metadata:
    :param p_transform:
    :param p_transform_augment:
    :param p_augment_sample:
    :param mm_center_location:
    :param mask_roi:
    :return:
    """
    patch_size = p_transform['patch_size']
    mm_patch_size = p_transform['mm_patch_size']

    # if p_augment_sample=None -> sample new params
    # if the transformation implies no augmentations then p_augment_sample remains None
    if not p_augment_sample and p_transform_augment:
        p_augment_sample = sample_augmentation_parameters(p_transform)

    # build scaling transformation
    original_size = data.shape[-2:]

    # scale the images such that they all have the same scale
    norm_scale_factor = (1. / pixel_spacing[0], 1. / pixel_spacing[1])
    mm_shape = tuple(int(float(d) * ps) for d, ps in zip(original_size, pixel_spacing))

    tform_normscale = build_rescale_transform(scaling_factor=norm_scale_factor,
                                              image_shape=original_size, target_shape=mm_shape)

    tform_shift_center, tform_shift_uncenter = build_shift_center_transform(image_shape=mm_shape,
                                                                            center_location=mm_center_location,
                                                                            patch_size=mm_patch_size)

    patch_scale_factor = (1. * mm_patch_size[0] / patch_size[0], 1. * mm_patch_size[1] / patch_size[1])

    tform_patch_scale = build_rescale_transform(patch_scale_factor, mm_patch_size, target_shape=patch_size)

    total_tform = tform_patch_scale + tform_shift_uncenter + tform_shift_center + tform_normscale

    # build random augmentation
    if p_augment_sample:
        augment_tform = build_augmentation_transform(rotation=p_augment_sample.rotation,
                                                     shear=p_augment_sample.shear,
                                                     translation=p_augment_sample.translation,
                                                     flip_x=p_augment_sample.flip_x,
                                                     flip_y=p_augment_sample.flip_y,
                                                     zoom=p_augment_sample.zoom)
        total_tform = tform_patch_scale + tform_shift_uncenter + augment_tform + tform_shift_center + tform_normscale

    # apply transformation to the slice
    out_data = fast_warp(data, total_tform, output_shape=patch_size)

    # apply transformation to ROI and mask the images
    segmentation_mask = np.ones_like(data)
    for a in annotations:
        center = a['center']
        radius_mm = a['diameter_mm'] / 2.
        radius = (int(radius_mm / pixel_spacing[0]),
                  int(radius_mm / pixel_spacing[1]))

        nodule_mask = make_roi_mask(original_size, center, radius, masked_value=0.1)
        segmentation_mask *= nodule_mask

    segmentation_mask = fast_warp(segmentation_mask, total_tform, output_shape=patch_size)

    return out_data, segmentation_mask


def make_roi_mask(img_shape, roi_center, roi_radii, shape='circle', masked_value=0.):
    if shape == 'circle':
        mask = np.ones(img_shape) * masked_value
        rr, cc = skimage.draw.ellipse(roi_center[0], roi_center[1], roi_radii[0], roi_radii[1], img_shape)
        mask[rr, cc] = 1.
    else:
        mask = np.ones(img_shape) * masked_value
        sx = slice(roi_center[0] - roi_radii[0], roi_center[0] + roi_radii[0])
        sy = slice(roi_center[1] - roi_radii[1], roi_center[1] + roi_radii[1])
        mask[sx, sy] = 1.
    return mask


tform_identity = skimage.transform.AffineTransform()


def fast_warp(img, tf, output_shape, mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params  # tf._matrix is
    return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)


def build_centering_transform(image_shape, target_shape=(50, 50)):
    rows, cols = image_shape
    trows, tcols = target_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))


def build_rescale_transform(scaling_factor, image_shape, target_shape):
    """
    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't
    100% correct, but it shouldn't matter much in practice.
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    tform_ds = skimage.transform.AffineTransform(scale=scaling_factor)

    # centering
    shift_x = cols / (2.0 * scaling_factor[0]) - tcols / 2.0
    shift_y = rows / (2.0 * scaling_factor[1]) - trows / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array(
        [image_shape[1], image_shape[0]]) / 2.0 - 0.5  # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter


def build_augmentation_transform(rotation=0, shear=0, translation=(0, 0), flip_x=False, flip_y=False, zoom=(1.0, 1.0)):
    if flip_x:
        shear += 180  # shear by 180 degrees is equivalent to flip along the X-axis
    if flip_y:
        shear += 180
        rotation += 180

    tform_augment = skimage.transform.AffineTransform(scale=(1. / zoom[0], 1. / zoom[1]), rotation=np.deg2rad(rotation),
                                                      shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def correct_orientation(data, metadata, roi_center, roi_radii):
    F = metadata["ImageOrientationPatient"].reshape((2, 3))
    f_1 = F[1, :]
    f_2 = F[0, :]
    y_e = np.array([0, 1, 0])
    if abs(np.dot(y_e, f_1)) >= abs(np.dot(y_e, f_2)):
        out_data = np.transpose(data, (0, 2, 1))
        out_roi_center = (roi_center[1], roi_center[0]) if roi_center else None
        out_roi_radii = (roi_radii[1], roi_radii[0]) if roi_radii else None
    else:
        out_data = data
        out_roi_center = roi_center
        out_roi_radii = roi_radii

    return out_data, out_roi_center, out_roi_radii


def build_shift_center_transform(image_shape, center_location, patch_size):
    """Shifts the center of the image to a given location.
    This function tries to include as much as possible of the image in the patch
    centered around the new center. If the patch arount the ideal center
    location doesn't fit within the image, we shift the center to the right so
    that it does.
    params in (i,j) coordinates !!!
    """
    if center_location[0] < 1. and center_location[1] < 1.:
        center_absolute_location = [
            center_location[0] * image_shape[0], center_location[1] * image_shape[1]]
    else:
        center_absolute_location = [center_location[0], center_location[1]]

    # Check for overlap at the edges
    center_absolute_location[0] = max(
        center_absolute_location[0], patch_size[0] / 2.0)
    center_absolute_location[1] = max(
        center_absolute_location[1], patch_size[1] / 2.0)

    center_absolute_location[0] = min(
        center_absolute_location[0], image_shape[0] - patch_size[0] / 2.0)

    center_absolute_location[1] = min(
        center_absolute_location[1], image_shape[1] - patch_size[1] / 2.0)

    # Check for overlap at both edges
    if patch_size[0] > image_shape[0]:
        center_absolute_location[0] = image_shape[0] / 2.0
    if patch_size[1] > image_shape[1]:
        center_absolute_location[1] = image_shape[1] / 2.0

    # Build transform
    new_center = np.array(center_absolute_location)
    translation_center = new_center - 0.5
    translation_uncenter = -np.array((patch_size[0] / 2.0, patch_size[1] / 2.0)) - 0.5
    return (
        skimage.transform.SimilarityTransform(translation=translation_center[::-1]),
        skimage.transform.SimilarityTransform(translation=translation_uncenter[::-1]))
