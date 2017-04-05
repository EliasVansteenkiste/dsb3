from collections import namedtuple
import numpy as np
import scipy.ndimage
import math
import utils_lung

MAX_HU = 400.
MIN_HU = -1000.
rng = np.random.RandomState(317070)




def hu2normHU(x):
    """
    Modifies input data
    :param x:
    :return:
    """
    x = (x - MIN_HU) / (MAX_HU - MIN_HU)
    x = np.clip(x, 0., 1., out=x)
    return x


def pixelnormHU(x):
    x = (x - MIN_HU) / (MAX_HU - MIN_HU)
    x = np.clip(x, 0., 1., out=x)
    return (x - 0.5) / 0.5


def histogram_equalization(x, hist=None, bins=None):
    # hist is a normalized histogram, which means that the sum of the counts has to be one
    if hist is None and bins is None:
        # For the case no target histogram is given
        bins = np.arange(-950,500,100)
        n_bins = bins.shape[0] -1
        hist = 1. * np.ones(n_bins) / n_bins
    elif hist is None or bins is None:
        raise
        
    assert(len(bins) == (len(hist)+1))

    # init our target array 
    z = np.empty(x.shape)

    # copy the values outside of the bins from the original
    z[x<=bins[0]] = x[x<=bins[0]] 
    z[x>=bins[-1]] = x[x>=bins[-1]] 

    inside_bins = np.logical_and(x>bins[0], x<bins[-1])

    n_bins = bins.shape[0] -1
    prev_percentile = 0
    for i in range(n_bins):
        target_count = hist[i]
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        new_percentile = prev_percentile + target_count*100
        low_orig = np.percentile(x[inside_bins], prev_percentile)
        if i == n_bins-1:
            high_orig = bins[-1]
        else:
            high_orig = np.percentile(x[inside_bins], new_percentile)

        prev_percentile = new_percentile

        elements_to_rescale = np.logical_and(x>=low_orig, x<high_orig)
        y = x[elements_to_rescale]
        y_r = (y - low_orig)/(high_orig-low_orig)*(upper_bound-lower_bound) + lower_bound
        print 'y_r', np.isnan(y_r).any()
        z[elements_to_rescale] = y_r

    return z

def get_rescale_params_hist_eq(x, hist=None, bins=None):
    # hist is a normalized histogram, which means that the sum of the counts has to be one
    if hist is None and bins is None:
        # For the case no target histogram is given
        bins = np.arange(-950,500,100)
        n_bins = bins.shape[0] -1
        hist = 1. * np.ones(n_bins) / n_bins
    elif hist is None or bins is None:
        raise
        
    assert(len(bins) == (len(hist)+1))

    inside_bins = np.logical_and(x>bins[0], x<bins[-1])

    n_bins = bins.shape[0] -1
    prev_percentile = 0
    original_borders = []
    for i in range(n_bins):
        target_count = hist[i]
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        new_percentile = prev_percentile + target_count*100
        low_orig = np.percentile(x[inside_bins], prev_percentile)
        original_borders.append(low_orig)
        prev_percentile = new_percentile
    original_borders.append(bins[-1])
        

    return bins, original_borders

def apply_hist_eq_patch(x, bins, original_borders):

    # init our target array 
    z = np.empty(x.shape)

    # if np.isnan(z).any():
    #     print '1 np.isnan(z).any()', np.isnan(z).any()

    # copy the values outside of the bins from the original
    z[x<=bins[0]] = x[x<=bins[0]] 
    z[x>=bins[-1]] = x[x>=bins[-1]] 
    # print 'x.shape', x.shape, x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
    # print 'np.sum(x<=bins[0])', np.sum(x<=bins[0])
    # print 'np.sum(x>=bins[-1])', np.sum(x>=bins[-1])

    # if np.isnan(z).any():
    #     print '2 np.isnan(z).any()', np.isnan(z).any()

    inside_bins = np.logical_and(x>bins[0], x<bins[-1])
    # print 'np.sum(inside_bins)', np.sum(inside_bins)

    n_total_elements_replaced = 0
    n_bins = bins.shape[0] -1
    for i in range(n_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        low_orig = original_borders[i]
        high_orig = original_borders[i+1]

        elements_to_rescale = np.logical_and(x>=low_orig, x<high_orig)
        n_total_elements_replaced += np.sum(elements_to_rescale)    
        # print 'np.sum(elements_to_rescale)', np.sum(elements_to_rescale)  
        y = x[elements_to_rescale]
        y_r = (y - low_orig)/(high_orig-low_orig)*(upper_bound-lower_bound) + lower_bound

        z[elements_to_rescale] = y_r

    #     if np.isnan(z).any():
    #         print 'np.isnan(z).any()', np.isnan(z).any()

    # print 'n_total_elements_replaced', n_total_elements_replaced
        
    return z


def sample_augmentation_parameters(transformation):
    shift_z = rng.uniform(*transformation.get('translation_range_z', [0., 0.]))
    shift_y = rng.uniform(*transformation.get('translation_range_y', [0., 0.]))
    shift_x = rng.uniform(*transformation.get('translation_range_x', [0., 0.]))
    translation = (shift_z, shift_y, shift_x)

    rotation_z = rng.uniform(*transformation.get('rotation_range_z', [0., 0.]))
    rotation_y = rng.uniform(*transformation.get('rotation_range_y', [0., 0.]))
    rotation_x = rng.uniform(*transformation.get('rotation_range_x', [0., 0.]))
    rotation = (rotation_z, rotation_y, rotation_x)

    return namedtuple('Params', ['translation', 'rotation'])(translation, rotation)


def transform_scan3d(data, pixel_spacing, p_transform,
                     luna_annotations=None,
                     luna_origin=None,
                     p_transform_augment=None,
                     world_coord_system=True,
                     lung_mask=None):
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

    if lung_mask is not None:
        lung_mask_out = apply_affine_transform(lung_mask, tf_total, order=1, output_shape=output_shape)
        lung_mask_out[lung_mask_out > 0.] = 1.
    if luna_annotations is not None:
        annotatations_out = []
        for zyxd in luna_annotations:
            zyx = np.array(zyxd[:3])
            voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing) if world_coord_system else zyx
            voxel_coords = np.append(voxel_coords, [1])
            voxel_coords_out = np.linalg.inv(tf_total).dot(voxel_coords)[:3]
            diameter_mm = zyxd[-1]
            diameter_out = diameter_mm * output_shape[1] / mm_patch_size[1] / out_pixel_spacing[1]
            zyxd_out = np.rint(np.append(voxel_coords_out, diameter_out))
            annotatations_out.append(zyxd_out)
        annotatations_out = np.asarray(annotatations_out)
        if lung_mask is None:
            return data_out, annotatations_out, tf_total
        else:
            return data_out, annotatations_out, tf_total, lung_mask_out

    if lung_mask is None:
        return data_out, tf_total
    else:
        return data_out, tf_total, lung_mask_out


def transform_patch3d(data, pixel_spacing, p_transform,
                      patch_center,
                      luna_origin,
                      luna_annotations=None,
                      p_transform_augment=None,
                      world_coord_system=True):
    mm_patch_size = np.asarray(p_transform['mm_patch_size'], dtype='float32')
    out_pixel_spacing = np.asarray(p_transform['pixel_spacing'])

    input_shape = np.asarray(data.shape)
    mm_shape = input_shape * pixel_spacing / out_pixel_spacing
    output_shape = p_transform['patch_size']

    zyx = np.array(patch_center[:3])
    voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing) if world_coord_system else zyx
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
        # print 'augmentation parameters', augment_params_sample
        tf_augment = affine_transform(translation=augment_params_sample.translation,
                                      rotation=augment_params_sample.rotation)
        tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_augment).dot(tf_shift_uncenter).dot(tf_output_scale)
    else:
        tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_shift_uncenter).dot(tf_output_scale)

    data_out = apply_affine_transform(data, tf_total, order=1, output_shape=output_shape)

    # transform patch annotations
    diameter_mm = patch_center[-1]
    diameter_out = diameter_mm * output_shape[1] / mm_patch_size[1] / out_pixel_spacing[1]
    voxel_coords = np.append(voxel_coords, [1])
    voxel_coords_out = np.linalg.inv(tf_total).dot(voxel_coords)[:3]
    patch_annotation_out = np.rint(np.append(voxel_coords_out, diameter_out))
    # print 'pathch_center_after_transform', patch_annotation_out

    if luna_annotations is not None:
        annotatations_out = []
        for zyxd in luna_annotations:
            zyx = np.array(zyxd[:3])
            voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing) if world_coord_system else zyx
            voxel_coords = np.append(voxel_coords, [1])
            voxel_coords_out = np.linalg.inv(tf_total).dot(voxel_coords)[:3]
            diameter_mm = zyxd[-1]
            diameter_out = diameter_mm * output_shape[1] / mm_patch_size[1] / out_pixel_spacing[1]
            zyxd_out = np.rint(np.append(voxel_coords_out, diameter_out))
            annotatations_out.append(zyxd_out)
        annotatations_out = np.asarray(annotatations_out)
        return data_out, patch_annotation_out, annotatations_out

    return data_out, patch_annotation_out


def transform_patch3d_ls(data, pixel_spacing, p_transform,
                      patch_center,
                      luna_origin,
                      p_transform_augment=None,
                      world_coord_system=True):
    mm_patch_size = np.asarray(p_transform['mm_patch_size'], dtype='float32')
    out_pixel_spacing = np.asarray(p_transform['pixel_spacing'])

    input_shape = np.asarray(data.shape)
    mm_shape = input_shape * pixel_spacing / out_pixel_spacing
    output_shape = p_transform['patch_size']

    zyx = np.array(patch_center[:3])
    # voxel_coords = utils_lung.world2voxel(zyx, luna_origin, pixel_spacing) if world_coord_system else zyx
    # voxel_coords_mm = voxel_coords * mm_shape / input_shape
    voxel_coords_mm = zyx * mm_shape / input_shape

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
        # print 'augmentation parameters', augment_params_sample
        tf_augment = affine_transform(translation=augment_params_sample.translation,
                                      rotation=augment_params_sample.rotation)
        tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_augment).dot(tf_shift_uncenter).dot(tf_output_scale)
    else:
        tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_shift_uncenter).dot(tf_output_scale)


    print 'data min,max', np.amin(data), np.amax(data)
    data_out = apply_affine_transform(data, tf_total, order=1, output_shape=output_shape)
    print 'data_out min,max', np.amin(data_out), np.amax(data_out)

    # transform patch annotations
    # voxel_coords = np.append(voxel_coords, [1])
    # voxel_coords_out = np.linalg.inv(tf_total).dot(voxel_coords)[:3]
    # patch_annotation_out = np.rint(voxel_coords_out)
    # print 'pathch_center_after_transform', patch_annotation_out

    return data_out #, patch_annotation_out


def transform_dsb_candidates(data, patch_centers, pixel_spacing, p_transform,
                             p_transform_augment=None):
    input_shape = np.asarray(data.shape)
    output_shape = np.asarray(p_transform['patch_size'])

    patches_out = []
    for zyxd in patch_centers:
        if -1 in zyxd:
            patch_out = np.zeros(output_shape)
        elif 'affine_tf' in p_transform and not p_transform['affine_tf']:
            assert(output_shape[0] == output_shape[1])
            assert(output_shape[0] == output_shape[2])

            zyx = np.round(np.array(zyxd[:3])).astype('int32')

            z_in = zyx[0] > output_shape[0]/2 and zyx[0] < input_shape[0]-output_shape[0]/2
            y_in = zyx[1] > output_shape[1]/2 and zyx[1] < input_shape[1]-output_shape[1]/2
            x_in = zyx[2] > output_shape[2]/2 and zyx[2] < input_shape[2]-output_shape[2]/2

            patch_inside_tensor = z_in and y_in and x_in

            if patch_inside_tensor:
                patch_out = data[zyx[0]-output_shape[0]/2:zyx[0]+output_shape[0]/2,
                                 zyx[1]-output_shape[1]/2:zyx[1]+output_shape[1]/2,
                                 zyx[2]-output_shape[2]/2:zyx[2]+output_shape[2]/2] 
            else:
                data_pad = np.empty((input_shape[0]+output_shape[0], 
                                     input_shape[1]+output_shape[1], 
                                     input_shape[2]+output_shape[2]))

                data_pad[0:output_shape[0]/2,:,:] = 0
                data_pad[output_shape[0]/2+input_shape[0]:,:,:] = 0

                data_pad[:,0:output_shape[1]/2,:] = 0
                data_pad[:,output_shape[1]/2+input_shape[1]:,:] = 0

                data_pad[:,:,0:output_shape[2]/2] = 0
                data_pad[:,:,output_shape[2]/2+input_shape[2]:] = 0

                data_pad[output_shape[0]/2:output_shape[0]/2+input_shape[0],
                         output_shape[1]/2:output_shape[1]/2+input_shape[1],
                         output_shape[2]/2:output_shape[2]/2+input_shape[2],] = data

                #too slow data_pad = np.lib.pad(data, output_shape[0], mode='constant', constant_values = MIN_HU)

                zyx_pad = zyx + output_shape/2
                patch_out = data_pad[zyx_pad[0]-output_shape[0]/2:zyx_pad[0]+output_shape[0]/2,
                                     zyx_pad[1]-output_shape[1]/2:zyx_pad[1]+output_shape[1]/2,
                                     zyx_pad[2]-output_shape[2]/2:zyx_pad[2]+output_shape[2]/2] 
        else:
            mm_patch_size = np.asarray(p_transform['mm_patch_size'], dtype='float32')
            out_pixel_spacing = np.asarray(p_transform['pixel_spacing'])
            mm_shape = input_shape * pixel_spacing / out_pixel_spacing

            zyx = np.array(zyxd[:3])
            zyx_mm = zyx * mm_shape / input_shape

            tf_mm_scale = affine_transform(scale=mm_shape / input_shape)
            tf_shift_center = affine_transform(translation=-zyx_mm)
            tf_shift_uncenter = affine_transform(translation=mm_patch_size / 2.)
            tf_output_scale = affine_transform(scale=output_shape / mm_patch_size)

            if p_transform_augment:
                augment_params_sample = sample_augmentation_parameters(p_transform_augment)
                tf_augment = affine_transform(translation=augment_params_sample.translation,
                                              rotation=augment_params_sample.rotation)
                tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_augment).dot(tf_shift_uncenter).dot(tf_output_scale)
            else:
                tf_total = tf_mm_scale.dot(tf_shift_center).dot(tf_shift_uncenter).dot(tf_output_scale)

            patch_out = apply_affine_transform(data, tf_total, order=p_transform['order'], output_shape=output_shape)
        
        patches_out.append(patch_out[None, :, :, :])
    return np.concatenate(patches_out, axis=0)


def build_dsb_can_heatmap(data, candidates, pixel_spacing, p_transform,
                             p_transform_augment=None):

    assert(candidates.shape[1]>3)

    mm_patch_size = np.asarray(p_transform['mm_patch_size'], dtype='float32')
    out_pixel_spacing = np.asarray(p_transform['pixel_spacing'])

    input_shape = np.asarray(data.shape)
    mm_shape = input_shape * pixel_spacing / out_pixel_spacing

    output_shape = p_transform['heatmap_size']
    max_shape = p_transform['max_shape']

    # Constructing heatmap
    heatmap = np.zeros(output_shape)
    max_dims = np.zeros(3)
    min_dims = 99999*np.ones(3)
    for can in candidates:
        value = can[-1]
        zyx = np.array(can[:3])
        zyx_mm = zyx * mm_shape / input_shape
        #only for analyse purpose
        for idx, d in enumerate(zyx_mm):
            if d>max_dims[idx]:
                max_dims[idx] = d
            if d<min_dims[idx]:
                min_dims[idx] = d
        zyx_hm = zyx_mm / max_shape * output_shape
        heatmap[zyx_hm.astype('int')] += value 

    # print 'max_dims', max_dims
    # print 'min_dims', min_dims
    # print 'heatmap max', np.amax(heatmap)
    # print 'heatmap min', np.amin(heatmap)

    # augmentation
    if p_transform_augment:
        augment_params_sample = sample_augmentation_parameters(p_transform_augment)
        tf_augment = affine_transform(translation=augment_params_sample.translation, rotation=augment_params_sample.rotation)
        heatmap = apply_affine_transform(heatmap, tf_augment, order=p_transform['heatmap_order'], output_shape=output_shape)

    heatmap = heatmap / p_transform['heatmap_norm']

    return heatmap


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
    elif shape == 'gauss':
        z, y, x = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
        distance = ((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
        mask = np.exp(- 1. * distance / (2 * radius ** 2))
        mask[(distance > 3 * radius ** 2)] = 0
    return mask


def make_3d_mask_from_annotations(img_shape, annotations, shape):
    mask = np.zeros(img_shape)
    for zyxd in annotations:
        mask += make_3d_mask(img_shape, zyxd[:3], zyxd[-1] / 2, shape)
    mask = np.clip(mask, 0., 1.)
    return mask


def make_gaussian_annotation(patch_annotation_tf, patch_size):
    radius = patch_annotation_tf[-1] / 2.
    zyx = patch_annotation_tf[:3]
    distance_z = (zyx[0] - np.arange(patch_size[0])) ** 2
    distance_y = (zyx[1] - np.arange(patch_size[1])) ** 2
    distance_x = (zyx[2] - np.arange(patch_size[2])) ** 2
    z_label = np.exp(- 1. * distance_z / (2 * radius ** 2))
    y_label = np.exp(- 1. * distance_y / (2 * radius ** 2))
    x_label = np.exp(- 1. * distance_x / (2 * radius ** 2))
    label = np.vstack((z_label, y_label, x_label))
    return label


def zmuv(x, mean, std):
    if mean is not None and std is not None:
        return (x - mean) / std
    else:
        return x


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

        matrix = mx.dot(my).dot(mz).dot(matrix)
    return matrix


def apply_affine_transform(_input, matrix, order=1, output_shape=None):
    # output.dot(T) + s = input
    T = matrix[:3, :3]
    s = matrix[:3, 3]
    return scipy.ndimage.interpolation.affine_transform(
        _input, matrix=T, offset=s, order=order, output_shape=output_shape)
