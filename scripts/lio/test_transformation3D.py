import preptools
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.transformation3D import affine_transform, apply_affine_transform, rescale_transform

PREVIEW_SHAPE = 600,600
cv2.namedWindow("Image", cv2.CV_WINDOW_AUTOSIZE)
cv2.moveWindow("Image", 1800, 300)
cv2.resizeWindow("Image", *PREVIEW_SHAPE)

input_shape = np.asarray((160., 512., 512.))
img = np.zeros(input_shape.astype("int"))
pixel_spacing = np.asarray([2., 0.7, 0.7])
output_shape =  np.asarray((100., 100., 100.))

img[:, 50:450, 250:450] = 100 #make cubish thing


norm_shape = input_shape*pixel_spacing
scale = output_shape / (input_shape * pixel_spacing)
fixed_real_shape = input_shape * pixel_spacing * np.min(scale)


shift_center = affine_transform(translation=-input_shape/2.)
normscale = affine_transform(scale=norm_shape/input_shape)
augment = affine_transform(translation=(0, 30, 0), rotation=(10, 0, 0))
fixedscale = affine_transform(scale=fixed_real_shape/norm_shape)
unshift_center = affine_transform(translation=fixed_real_shape/2.)

m = shift_center.dot(normscale).dot(augment).dot(fixedscale).dot(unshift_center)



_img = cv2.resize(img[img.shape[0]//2], PREVIEW_SHAPE, interpolation=cv2.INTER_NEAREST)
cv2.imshow("Image", _img)
cv2.waitKey(0)

img = apply_affine_transform(img, m, order=1, output_shape=output_shape.astype("int"))


print img.shape
_img = cv2.resize(img[img.shape[0]//2], PREVIEW_SHAPE, interpolation=cv2.INTER_NEAREST)
cv2.imshow("Image", _img)
cv2.waitKey(0)












# # preptools.plot_3d(a, theshold=1, cut=False)
# mc = affine_transform(translation=-np.array(a.shape)/2.)
# muc = affine_transform(translation=np.array(a.shape)/2.)
# mr = affine_transform(rotation=(50, 0, 0))
# mf = affine_transform(scale=(1,-1,1))
# # mt = affine_transform(translation=(-75,-75,-75))
# # m = ms.dot(mt)
# # m = ms
# # print m, m[:3,3]
# # m =
#
# ms = rescale_transform(input_shape=a.shape, output_shape=output_shape)
#
#
# m = ms.dot(muc).dot(mf).dot(mr).dot(mc)
#
#
# # normscale = affine_transform(scale=(1,-1,1), origin=np.array(a.shape)/2.)
# # augment = affine_transform(translation=(1,1,1), origin=np.array(a.shape)/2.)
# # rescale = rescale_transform(input_shape=a.shape, output_shape=output_shape)
# #
# # m = augment.dot(normscale).dot(rescale)
#
# print m
# # for img  in a:
# #     img = cv2.resize(img, PREVIEW_SHAPE, interpolation=cv2.INTER_NEAREST)
# #     cv2.imshow("Image", img)
# #     cv2.waitKey(10)
#
# img = cv2.resize(a[a.shape[0]//2], PREVIEW_SHAPE, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# a = apply_affine_transform(a, m, order=3, output_shape=output_shape)
# print a.shape
# img = cv2.resize(a[a.shape[0]//2], PREVIEW_SHAPE, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
#
#
# # plt.imshow(a[0], cmap='gray')
# # plt.show()
# # preptools.plot_3d_mpl(a, threshold=1)
# # preptools.plot_3d(a, threshold=1, cut=False)
# # def show_img(img, wait=0, resize=True, resize_shape=PREVIEW_SHAPE):
# #     if img.max() > 1: img = img.astype("uint8")
# #     if resize: img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_NEAREST)
# #     cv2.imshow("Image", img)
# #     cv2.waitKey(wait)