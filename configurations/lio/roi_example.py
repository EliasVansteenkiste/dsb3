import numpy as np
from itertools import product
from functools import partial

from configurations.jonas import valid
from scripts.elias.blob import blob_dog

from application.stage1 import Stage1DataLoader
from interfaces.data_loader import VALIDATION, TRAINING, TEST, TRAIN, INPUT
from application.preprocessors.dicom_to_HU import DicomToHU
from utils.transformation_3d import affine_transform, apply_affine_transform
from interfaces.preprocess import ZMUV


model = valid

IMAGE_SIZE = 128
patch_shape = IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE  # in pixels
norm_patch_shape = IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE  # in mms

replace_input_tags = {"luna:3d": "stage1:3d"}

preprocessors = [DicomToHU(tags=["stage1:3d"])]
postpreprocessors = [ZMUV("stage1:3d", bias =  -648.59027, std = 679.21021)] #lol

data_loader= Stage1DataLoader(
    sets=[TRAINING, VALIDATION],
    preprocessors=preprocessors,
    epochs=1,
    multiprocess=False,
    crash_on_exception=True)

batch_size = 1

def patch_generator(sample, segmentation_shape):
    for prep in preprocessors: prep.process(sample)

    data = sample[INPUT]["stage1:3d"]
    spacing = sample[INPUT]["stage1:pixelspacing"]

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

    for x,y,z in product(range(patch_count[0]), range(patch_count[1]), range(patch_count[2])):

        offset = np.array([stride[0]*x, stride[1]*y, stride[2]*z], np.float)
        print (x*patch_count[1]*patch_count[2] + y*patch_count[2] +z), "/", np.prod(patch_count)

        shift_center = affine_transform(translation=-input_shape / 2. - 0.5)
        normscale = affine_transform(scale=norm_shape / input_shape)
        offset_patch = affine_transform(translation=norm_shape/2.-0.5-offset+mm_patch_shape - segmentation_shape)
        patchscale = affine_transform(scale=_patch_shape / norm_shape)
        unshift_center = affine_transform(translation=output_shape / 2. - 0.5)
        matrix = shift_center.dot(normscale).dot(offset_patch).dot(patchscale).dot(unshift_center)
        output = apply_affine_transform(data, matrix, output_shape=output_shape.astype(np.int))

        patch = {}
        patch["stage1:3d"] = output
        patch["offset"] = offset
        yield patch


def extract_nodules(pred, patch):
    try:
        rois = blob_dog(pred, min_sigma=1.2, max_sigma=35, threshold=0.1)
    except:
        print "blob_dog failed"
        return None
    if rois.shape[0] > 0:
        rois = rois[:, :3] #ignore diameter
        rois += patch["offset"][None,:]
    else: return None
    # print rois
    #local to global roi
    # rois += patch["offset"]
    return rois


