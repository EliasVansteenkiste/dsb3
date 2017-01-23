import numpy as np
import math
import scipy.ndimage


def apply_affine_transform(_input, matrix, order=1, output_shape=None):
    # output.dot(T) + s = input
    T = matrix[:3, :3]
    s = matrix[:3, 3]
    return scipy.ndimage.interpolation.affine_transform(
        _input, T, offset=s, order=order, output_shape=output_shape)

def affine_transform(scale=None, rotation=None, translation=None, shear=None, reflection=None):
    """
    rotation and shear in degrees
    """
    matrix = np.eye(4)

    if not translation is None:
        matrix[:3, 3] = -np.asarray(translation, np.float)

    if not reflection is None:
        reflection = -np.asarray(reflection, np.float)*2+1
        if scale is None: scale = 1.

    if not scale is None:
        if reflection is None: reflection = 1.
        scale = reflection/np.asarray(scale, np.float)
        matrix[0,0] = scale[0]
        matrix[1,1] = scale[1]
        matrix[2,2] = scale[2]

    if not shear is None:
        if len(shear) != 6 and len(shear) != 3: raise Exception("shear should be of size 6 or 3")
        shear = -np.asarray(shear, np.float)
        shear = map(math.radians, shear)
        shear = map(math.tan, shear)

        matrix[0, 1] = shear[0] #x
        matrix[0, 2] = shear[1] #y
        matrix[1, 2] = shear[2] #z

        if len(shear) == 6:
            matrix[1, 0] = shear[3]
            matrix[2, 0] = shear[4]
            matrix[2, 1] = shear[5]

    if rotation is None: return matrix

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

    return matrix.dot(mx).dot(my).dot(mz)
