import numpy as np
import math
import scipy.ndimage


def apply_affine_transform(_input, matrix, order=1, output_shape=None, cval=0):
    # output.dot(T) + s = input
    T = matrix[:3, :3]
    s = matrix[:3, 3]
    return scipy.ndimage.interpolation.affine_transform(
        _input, T, offset=s, order=order, output_shape=output_shape, cval=cval)

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

        matrix[0, 1] = shear[2] #x
        matrix[0, 2] = shear[1] #y
        matrix[1, 2] = shear[0] #z

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


def test_transformation_3d():
    import numpy as np

    input_shape = np.asarray((160., 512., 512.))
    img = np.ones(input_shape.astype("int"))*50
    pixel_spacing = np.asarray([0.7, .7, .7])
    output_shape =  np.asarray((100., 100., 100.))
    norm_patch_size = np.asarray((112, 300 ,300), np.float)

    # img[:, 50:450, 250:450] = 100 #make cubish thing
    s = np.asarray(img.shape)//2
    img[s[0]-50:s[0]+50, s[1]-50:s[1]+200, s[2]-50:s[2]+50] = 100 #make cubish thing


    norm_shape = input_shape*pixel_spacing
    print norm_shape
    patch_shape = norm_shape * np.min(output_shape/norm_patch_size)
    # patch_shape = norm_shape * output_shape/norm_patch_size


    shift_center = affine_transform(translation=-input_shape/2.-0.5)
    normscale = affine_transform(scale=norm_shape/input_shape)
    augment = affine_transform(reflection=(0,1,0))
    patchscale = affine_transform(scale=patch_shape/norm_shape)
    unshift_center = affine_transform(translation=output_shape/2.-0.5)

    matrix = shift_center.dot(normscale).dot(augment).dot(patchscale).dot(unshift_center)

    img_trans = apply_affine_transform(img, matrix, order=1, output_shape=output_shape.astype("int"))


    import utils.plt
    utils.plt.show_compare(img, img_trans)


if __name__ == '__main__':
    test_transformation_3d()