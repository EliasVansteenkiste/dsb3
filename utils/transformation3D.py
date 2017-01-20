"""
Example:

rotate = affine_transform(rotation=(50, 30, 20), origin=(50,50,50))
translate = affine_transform(translation=(5,10,10))
flip = affine_transform(scale=(1,-1,1)) #flip about Y axis

matrix = flip.dot(translate).dot(rotate) #rotation followed by translation followed by flip

img = apply_affine_transform(img, matrix)

"""
import numpy as np
import math
import scipy.ndimage


def apply_affine_transform(_input, matrix, order=1, output_shape=None):
    return scipy.ndimage.interpolation.affine_transform(
        _input, matrix[:3,:3], offset=matrix[:3, 3], order=order, output_shape=output_shape)


def rescale_transform(input_shape, output_shape):
    input_shape = np.asarray(input_shape, np.float)
    output_shape = np.asarray(output_shape, np.float)
    scale = output_shape/input_shape
    translation = (input_shape - output_shape) / 2. / scale
    print scale, translation

    tfscale = affine_transform(scale=scale, origin=input_shape/2.)
    tftrans = affine_transform(translation=translation)
    return tftrans.dot(tfscale)
    # if output_shape is not None:
    #     translation = -(np.asarray(_input.shape) - np.asarray(output_shape))/2.
    #     translation *= abs(matrix[0,0]), abs(matrix[1,1]), abs(matrix[2,2])
    #     offset += translation

def affine_transform(scale=None, rotation=None, translation=None, shear=None, origin=None, output_shape=None):
    """
    rotation and shear in degrees
    """
    matrix = np.eye(4)

    if not translation is None:
        matrix[:3, 3] = -np.asarray(translation, np.float)

    if not scale is None:
        scale = 1./np.asarray(scale, np.float)
        matrix[0,0] = scale[0]
        matrix[1,1] = scale[1]
        matrix[2,2] = scale[2]

    if not shear is None:
        if len(shear) != 6: raise Exception("shear should be of size 6")
        shear = -np.asarray(shear, np.float)
        shear = map(math.radians, shear)
        shear = map(math.tan, shear)
        matrix[0, 1] = shear[0]
        matrix[0, 2] = shear[1]
        matrix[1, 0] = shear[2]
        matrix[1, 2] = shear[3]
        matrix[2, 0] = shear[4]
        matrix[2, 1] = shear[5]

    if rotation is None:
        if not origin is None:
            origin = np.asarray(origin)
            shift = affine_transform(translation=-origin)
            unshift = affine_transform(translation=origin)
            return shift.dot(matrix).dot(unshift)
        else:
            return matrix

    # ROTATION

    if len(rotation) != 3: raise Exception("rotation should be of size 3")
    rotation = -np.asarray(rotation, np.float)
    rotation = map(math.radians, rotation)
    cos = map(math.cos, rotation)
    sin = map(math.sin, rotation)
    mx = np.eye(4)
    mx[1, 1] = cos[0]
    mx[2, 1] = sin[0]
    mx[1, 2] = -sin[0]
    mx[2, 2] = cos[0]

    my = np.eye(4)
    my[0, 0] = cos[1]
    my[0, 2] = sin[1]
    my[2, 0] = -sin[1]
    my[2, 2] = cos[1]

    mz = np.eye(4)
    mz[0, 0] = cos[2]
    mz[1, 0] = sin[2]
    mz[0, 1] = -sin[2]
    mz[1, 1] = cos[2]

    if not origin is None:
        shift = affine_transform(translation=-np.asarray(origin))
        unshift = affine_transform(translation=np.asarray(origin))
        return shift.dot(matrix).dot(mx).dot(my).dot(mz).dot(unshift)
    else:
        return matrix.dot(mx).dot(my).dot(mz)
