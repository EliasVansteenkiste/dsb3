import numpy as np
import matplotlib.pyplot as plt

from utils.transformation_3d import affine_transform, apply_affine_transform


input_shape = np.asarray((160., 512., 512.))
img = np.ones(input_shape.astype("int"))*50
pixel_spacing = np.asarray([0.7, .7, .7])
output_shape =  np.asarray((100., 100., 100.))
norm_patch_size = np.asarray((112, 300 ,300), np.float)

# img[:, 50:450, 250:450] = 100 #make cubish thing
s = np.asarray(img.shape)//2
img[s[0]-50:s[0]+50, s[1]-50:s[1]+50, s[2]-50:s[2]+50] = 100 #make cubish thing


norm_shape = input_shape*pixel_spacing
print norm_shape
patch_shape = norm_shape * np.min(output_shape/norm_patch_size)
# patch_shape = norm_shape * output_shape/norm_patch_size


shift_center = affine_transform(translation=-input_shape/2.-0.5)
normscale = affine_transform(scale=norm_shape/input_shape)
augment = affine_transform(scale=(), shear=(10, 10, 0))
patchscale = affine_transform(scale=patch_shape/norm_shape)
unshift_center = affine_transform(translation=output_shape/2.-0.5)

matrix = shift_center.dot(normscale).dot(augment).dot(patchscale).dot(unshift_center)

img_trans = apply_affine_transform(img, matrix, order=1, output_shape=output_shape.astype("int"))


plt.close('all')
fig, ax = plt.subplots(1,4)
ax[0].imshow(img[img.shape[0]//2]/100., cmap="gray")
ax[1].imshow(img_trans[img_trans.shape[0]//2]/100., cmap="gray")
ax[2].imshow(img_trans[:, img_trans.shape[1]//2]/100., cmap="gray")
ax[3].imshow(img_trans[:, :, img_trans.shape[2]//2]/100., cmap="gray")
plt.show()