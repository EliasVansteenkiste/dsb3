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
    T = matrix[:3, :3]
    s = matrix[:3, 3]
    print T, s
    return scipy.ndimage.interpolation.affine_transform(
        _input, matrix=T, offset=s, order=order, output_shape=output_shape)


def affine_transform(scale=None, rotation=None, translation=None, shear=None, origin=None, output_shape=None):
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

        mx = np.eye(4)
        mx[1, 1] = cos[0]
        mx[2, 1] = sin[0]
        mx[1, 2] = -sin[0]
        mx[2, 2] = cos[0]

        my = np.eye(4)
        my[0, 0] = cos[1]
        my[0, 2] = -sin[1]
        my[2, 0] = sin[1]
        my[2, 2] = cos[1]

        mz = np.eye(4)
        mz[0, 0] = cos[2]
        mz[0, 1] = sin[2]
        mz[1, 0] = -sin[2]
        mz[1, 1] = cos[2]

        matrix = matrix.dot(mx).dot(my).dot(mz)

    return matrix
