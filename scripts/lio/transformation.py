import numpy as np
import math


def make_tf_matrix(scale=None, rotation=None, translation=None):
    if scale is None: scale = (1,1,1)
    if translation is None: translation = (0,0,0)

    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(translation)
    matrix[0,0] = scale[0]
    matrix[1,1] = scale[1]
    matrix[2,2] = scale[2]

    if rotation is None: return matrix

    # ROTATION

    if len(rotation) != 3: raise "rotation should be a tuple of size 3"
    cosx = math.cos(rotation[0])
    sinx = math.sin(rotation[0])
    mx = np.eye(4)
    mx[1, 1] = cosx
    mx[2, 1] = sinx
    mx[1, 2] = -sinx
    mx[2, 2] = cosx

    cosy = math.cos(rotation[1])
    siny = math.sin(rotation[1])
    my = np.eye(4)
    my[0, 0] = cosy
    my[0, 2] = siny
    my[2, 0] = -siny
    my[2, 2] = cosy

    cosz = math.cos(rotation[2])
    sinz = math.sin(rotation[2])
    mz = np.eye(4)
    mz[0, 0] = cosz
    mz[1, 0] = sinz
    mz[0, 1] = -sinz
    mz[1, 1] = cosz

    # Rotx followed by Roty followed by Rotz followed by TransScale
    return np.dot(np.dot(np.dot(matrix, mz), my), mx)


if __name__ == '__main__':
    import preptools
    import scipy.ndimage

    a = np.zeros((100, 100, 100))

    a[25:75, 25:75, 25:75] = 100
    print a
    # preptools.plot_3d(a, theshold=1, cut=False)
    mt = make_tf_matrix(rotation=(0.1,0.1,0.1))
    # mr = make_tf_matrix(scale=(1,1,1), rotation=(math.pi/4, 0, 0))
    # mi = make_tf_matrix(translation=np.array(a.shape) / 2.)
    # m = np.dot(np.dot(mi, mr), mt)
    # m = mt.dot(mr.dot(mi))
    # print m
    m = mt

    a = scipy.ndimage.interpolation.affine_transform(a, m[:3,:3], order=3, offset=m[3,:3])
    print a.shape
    preptools.plot_3d(a, theshold=10, cut=False)

