import numpy as np
from itertools import product
from functools import partial

from configurations.jonas import valid, ira_config_2
from scripts.elias.blob import blob_dog
from application.luna import LunaDataLoader
from application.stage1 import Stage1DataLoader
from interfaces.data_loader import VALIDATION, TRAINING, TEST, TRAIN, INPUT
from application.preprocessors.dicom_to_HU import DicomToHU
from utils.transformation_3d import affine_transform, apply_affine_transform
from interfaces.preprocess import ZMUV


<<<<<<< HEAD
plot = False
multiprocess = True
=======
plot = True
>>>>>>> 30a53442643d68af8b91f38431a01f0dab88c8be

model = valid
tag = "luna:"
extra_tags=[tag+"pixelspacing"]

<<<<<<< HEAD
IMAGE_SIZE = 160
=======
IMAGE_SIZE = 180
>>>>>>> 30a53442643d68af8b91f38431a01f0dab88c8be
patch_shape = IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE  # in pixels
norm_patch_shape = IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE  # in mms

replace_input_tags = {"luna:3d": tag+"3d"}

preprocessors = []
postpreprocessors = [ZMUV(tag+"3d", bias =  -648.59027, std = 679.21021)]

data_loader= LunaDataLoader(
    sets=[TRAINING, VALIDATION],
    preprocessors=preprocessors,
    epochs=1,
    multiprocess=False,
    crash_on_exception=True)

batch_size = 1
stride = None
patch_count = None
norm_shape = None

def patch_generator(sample, segmentation_shape):
    global patch_count, stride, norm_shape

    for prep in preprocessors: prep.process(sample)

    data = sample[INPUT][tag+"3d"]
    spacing = sample[INPUT][tag+"pixelspacing"]

    input_shape = np.asarray(data.shape, np.float)
    pixel_spacing = np.asarray(spacing, np.float)
    output_shape = np.asarray(patch_shape, np.float)
    mm_patch_shape = np.asarray(norm_patch_shape, np.float)
    stride = np.asarray(segmentation_shape, np.float) * mm_patch_shape / output_shape

    norm_shape = input_shape * pixel_spacing
    _patch_shape = norm_shape * output_shape / mm_patch_shape

    patch_count = np.ceil(norm_shape / stride).astype("int")
    print "patch_count", patch_count
    print "stride", stride
    print spacing
<<<<<<< HEAD
    print norm_shape
=======
>>>>>>> 30a53442643d68af8b91f38431a01f0dab88c8be

    for x,y,z in product(range(patch_count[0]), range(patch_count[1]), range(patch_count[2])):

        offset = np.array([stride[0]*x, stride[1]*y, stride[2]*z], np.float)
        print (x*patch_count[1]*patch_count[2] + y*patch_count[2] +z), "/", np.prod(patch_count), (x,y,z)

        shift_center = affine_transform(translation=-(input_shape / 2. - 0.5))
        normscale = affine_transform(scale=norm_shape / input_shape)
        offset_patch = affine_transform(translation=norm_shape/2. - 0.5 - offset-(stride/2.0-0.5))# - (mm_patch_shape - segmentation_shape)*norm_shape/_patch_shape -segmentation_shape*norm_shape/_patch_shape/2.)
        patchscale = affine_transform(scale=_patch_shape / norm_shape)
        unshift_center = affine_transform(translation=output_shape / 2. - 0.5)
        matrix = shift_center.dot(normscale).dot(offset_patch).dot(patchscale).dot(unshift_center)
        output = apply_affine_transform(data, matrix, output_shape=output_shape.astype(np.int))


        patch = {}
        patch[tag+"3d"] = output
        patch["offset"] = offset
        s = {INPUT: patch}
        for prep in postpreprocessors: prep.process(s)
        yield patch


def glue_patches(p):
    global patch_count, stride, norm_shape

    preds = []
    for x in range(patch_count[0]):
        preds_y = []
        for y in range(patch_count[1]):
            ofs = y * patch_count[2] + x * patch_count[2] * patch_count[1]
            preds_z = np.concatenate(p[ofs:ofs + patch_count[2]], axis=2)
            preds_y.append(preds_z)
        preds_y = np.concatenate(preds_y, axis=1)
        preds.append(preds_y)

    preds = np.concatenate(preds, axis=0)
    preds = preds[:int(round(norm_shape[0])), :int(round(norm_shape[1])), :int(round(norm_shape[2]))]
    return preds


def extract_nodules(pred):
    try:
        rois = blob_dog(pred, min_sigma=1, max_sigma=15, threshold=0.1)
    except:
        print "blob_dog failed"
        return None
    if rois.shape[0] > 0:
        rois = rois[:, :3] #ignore diameter
        # rois += patch["offset"][None,:]
    else: return None
    #local to global roi
    # rois += patch["offset"]
    return rois


