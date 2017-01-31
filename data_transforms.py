from collections import namedtuple
import numpy as np
import skimage.transform
import skimage.draw
from configuration import config
import skimage.exposure, skimage.filters
import scipy.ndimage
import math
import utils_lung

MAX_HU = 400.
MIN_HU = -1000.


def ct2normHU(x, metadata):
    """
    modifies input data
    :param x:
    :param metadata:
    :return:
    """
    x[x < 0.] = 0.
    x = metadata['RescaleSlope'] * x + metadata['RescaleIntercept']
    x = (x - MIN_HU) / (MAX_HU - MIN_HU)
    x[x < 0.] = 0.
    x[x > 1.] = 1.
    return x


def hu2normHU(x):
    """
    Modifies input data
    :param x:
    :return:
    """
    x = (x - MIN_HU) / (MAX_HU - MIN_HU)
    x[x < 0.] = 0.
    x[x > 1.] = 1.
    return x


def sample_augmentation_parameters(transformation):
    shift_z = config().rng.uniform(*transformation.get('translation_range_z', [0., 0.]))
    shift_y = config().rng.uniform(*transformation.get('translation_range_y', [0., 0.]))
    shift_x = config().rng.uniform(*transformation.get('translation_range_x', [0., 0.]))
    translation = (shift_z, shift_y, shift_x)

    rotation_z = config().rng.uniform(*transformation.get('rotation_range_z', [0., 0.]))
    rotation_y = config().rng.uniform(*transformation.get('rotation_range_y', [0., 0.]))
    rotation_x = config().rng.uniform(*transformation.get('rotation_range_x', [0., 0.]))
    rotation = (rotation_z, rotation_y, rotation_x)

    return namedtuple('Params', ['translation', 'rotation'])(translation, rotation)


def transform_scan3d(data, pixel_spacing, p_transform,
                     luna_annotations=None,
                     luna_origin=None,
                     p_transform_augment=None):
    mm_patch_size = np.asarray(p_transform['mm_patch_size'], dtype='float32')
    out_pixel_spacing = np.asarray(p_transform['pixel_spacing'])

    input_shape = np.asarray(data.shape)
    mm_shape = input_shape * pixel_spacing / out_pixel_spacing
    output_shape = p_transform['patch_size']

    # here we give parameters to affine transform as if it's T in
    # output = T.dot(input)
    # https://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
    # but the affine_transform() makes it reversed for scipy
    tf_mm_scale = affine_transform(scale=mm_shape / input_shape)
    tf_shift_center = affine_transform(translation=-mm_shape / 2.)

    tf_shift_uncenter = affine_transform(translation=mm_patch_size / 2.)
    tf_output_scale = affine_transform(scale=output_shape / mm_patch_size)

    if p_transform_augment:
        augment_params_sample = sample_augmentation_parameters(p_transform_augment)
        tf_augment = affine_transform(translation=augment_params_sample.translation,
                                      rotation=augment_params_sample.rotation)
        tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_augment).dot(tf_shift_uncenter).dot(tf_output_scale)
    else:
        tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_shift_uncenter).dot(tf_output_scale)

    data_out = apply_affine_transform(data, tf_total, order=1, output_shape=output_shape)

    if luna_annotations is not None:
        annotatations_out = []
        for zyxd in luna_annotations:
            zyx = np.array(zyxd[:3])
            voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing)
            voxel_coords = np.append(voxel_coords, [1])
            voxel_coords_out = np.linalg.inv(tf_total).dot(voxel_coords)[:3]
            diameter_mm = zyxd[-1]
            diameter_out = diameter_mm * output_shape[1] / mm_patch_size[1]
            zyxd_out = np.rint(np.append(voxel_coords_out, diameter_out))
            annotatations_out.append(zyxd_out)
        return data_out, annotatations_out

    return data_out


def transform_patch3d(data, pixel_spacing, p_transform,
                      patch_center,
                      luna_origin,
                      luna_annotations=None,
                      p_transform_augment=None):
    mm_patch_size = np.asarray(p_transform['mm_patch_size'], dtype='float32')
    out_pixel_spacing = np.asarray(p_transform['pixel_spacing'])

    input_shape = np.asarray(data.shape)
    mm_shape = input_shape * pixel_spacing / out_pixel_spacing
    output_shape = p_transform['patch_size']

    zyx = np.array(patch_center[:3])
    voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing)
    voxel_coords_mm = voxel_coords * mm_shape / input_shape

    # here we give parameters to affine transform as if it's T in
    # output = T.dot(input)
    # https://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
    # but the affine_transform() makes it reversed for scipy
    tf_mm_scale = affine_transform(scale=mm_shape / input_shape)
    tf_shift_center = affine_transform(translation=-voxel_coords_mm)

    tf_shift_uncenter = affine_transform(translation=mm_patch_size / 2.)
    tf_output_scale = affine_transform(scale=output_shape / mm_patch_size)

    if p_transform_augment:
        augment_params_sample = sample_augmentation_parameters(p_transform_augment)
        tf_augment = affine_transform(translation=augment_params_sample.translation,
                                      rotation=augment_params_sample.rotation)
        tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_augment).dot(tf_shift_uncenter).dot(tf_output_scale)
    else:
        tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_shift_uncenter).dot(tf_output_scale)

    data_out = apply_affine_transform(data, tf_total, order=1, output_shape=output_shape)

    # transform patch annotations
    diameter_mm = patch_center[-1]
    diameter_out = diameter_mm * output_shape[1] / mm_patch_size[1]
    voxel_coords = np.append(voxel_coords, [1])
    voxel_coords_out = np.linalg.inv(tf_total).dot(voxel_coords)[:3]
    patch_annotation_out = np.rint(np.append(voxel_coords_out, diameter_out))

    if luna_annotations is not None:
        annotatations_out = []
        for zyxd in luna_annotations:
            zyx = np.array(zyxd[:3])
            voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing)
            voxel_coords = np.append(voxel_coords, [1])
            voxel_coords_out = np.linalg.inv(tf_total).dot(voxel_coords)[:3]
            diameter_mm = zyxd[-1]
            diameter_out = diameter_mm * output_shape[1] / mm_patch_size[1]
            zyxd_out = np.rint(np.append(voxel_coords_out, diameter_out))
            annotatations_out.append(zyxd_out)
        return data_out, patch_annotation_out, annotatations_out

    return data_out, patch_annotation_out


def luna_transform_slice(data, pixel_spacing, p_transform,
                         luna_origin,
                         luna_annotations=None,
                         p_transform_augment=None):
    patch_size = p_transform['patch_size']
    mm_patch_size = p_transform['mm_patch_size']

    # build scaling transformation
    original_size = data.shape[-2:]

    # scale the images such that they all have the same scale
    norm_scale_factor = (1. / pixel_spacing[-2], 1. / pixel_spacing[-1])
    mm_shape = tuple(int(float(d) * ps) for d, ps in zip(original_size, pixel_spacing))

    tform_normscale = build_rescale_transform(scaling_factor=norm_scale_factor,
                                              image_shape=original_size, target_shape=mm_shape)

    tform_shift_center, tform_shift_uncenter = build_shift_center_transform(image_shape=mm_shape,
                                                                            center_location=(0.5, 0.5),
                                                                            patch_size=mm_patch_size)

    patch_scale_factor = (1. * mm_patch_size[0] / patch_size[0], 1. * mm_patch_size[1] / patch_size[1])

    tform_patch_scale = build_rescale_transform(patch_scale_factor, mm_patch_size, target_shape=patch_size)

    total_tform = tform_patch_scale + tform_shift_uncenter + tform_shift_center + tform_normscale

    # build random augmentation
    if p_transform_augment is not None:
        p_augment_sample = sample_augmentation_parameters(p_transform)
        augment_tform = build_augmentation_transform(rotation=p_augment_sample.rotation,
                                                     translation=p_augment_sample.translation)
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

        nodule_mask = make_2d_mask(original_size, center, radius, masked_value=0.1)
        segmentation_mask *= nodule_mask

    segmentation_mask = fast_warp(segmentation_mask, total_tform, output_shape=patch_size)

    return out_data, segmentation_mask


def make_2d_mask(img_shape, roi_center, roi_radii, shape='circle', masked_value=0.):
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


def make_3d_mask(img_shape, center, radius, shape='sphere'):
    mask = np.zeros(img_shape)
    radius = np.rint(radius)
    center = np.rint(center)
    sz = np.arange(int(max(center[0] - radius, 0)), int(max(min(center[0] + radius + 1, img_shape[0]), 0)))
    sy = np.arange(int(max(center[1] - radius, 0)), int(max(min(center[1] + radius + 1, img_shape[1]), 0)))
    sx = np.arange(int(max(center[2] - radius, 0)), int(max(min(center[2] + radius + 1, img_shape[2]), 0)))
    sz, sy, sx = np.meshgrid(sz, sy, sx)
    if shape == 'cube':
        mask[sz, sy, sx] = 1.
    elif shape == 'sphere':
        distance2 = ((center[0] - sz) ** 2
                     + (center[1] - sy) ** 2
                     + (center[2] - sx) ** 2)
        distance_matrix = np.ones_like(mask) * np.inf
        distance_matrix[sz, sy, sx] = distance2
        mask[(distance_matrix <= radius ** 2)] = 1
        # z, y, x = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
        # distance2 = ((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
        # mask[(distance2 <= radius ** 2)] = 1
    return mask


def make_3d_mask_from_annotations(img_shape, annotations, shape):
    mask = np.zeros(img_shape)
    for zyxd in annotations:
        mask += make_3d_mask(img_shape, zyxd[:3], zyxd[-1] / 2, shape)
    mask[mask > 0] = 1.
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


def affine_transform(scale=None, rotation=None, translation=None):
    """
    rotation and shear in degrees
    """
    matrix = np.eye(4)

    if translation is not None:
        matrix[:3, 3] = -np.asarray(translation, np.float)

    if scale is not None:
        matrix[0, 0] = 1. / scale[0]
        matrix[1, 1] = 1. / scale[1]
        matrix[2, 2] = 1. / scale[2]

    if rotation is not None:
        rotation = np.asarray(rotation, np.float)
        rotation = map(math.radians, rotation)
        cos = map(math.cos, rotation)
        sin = map(math.sin, rotation)

        mz = np.eye(4)
        mz[1, 1] = cos[0]
        mz[2, 1] = sin[0]
        mz[1, 2] = -sin[0]
        mz[2, 2] = cos[0]

        my = np.eye(4)
        my[0, 0] = cos[1]
        my[0, 2] = -sin[1]
        my[2, 0] = sin[1]
        my[2, 2] = cos[1]

        mx = np.eye(4)
        mx[0, 0] = cos[2]
        mx[0, 1] = sin[2]
        mx[1, 0] = -sin[2]
        mx[1, 1] = cos[2]

        matrix = matrix.dot(mx).dot(my).dot(mz)

    return matrix


def apply_affine_transform(_input, matrix, order=1, output_shape=None):
    # output.dot(T) + s = input
    T = matrix[:3, :3]
    s = matrix[:3, 3]
    return scipy.ndimage.interpolation.affine_transform(
        _input, matrix=T, offset=s, order=order, output_shape=output_shape)
