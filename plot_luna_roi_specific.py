
import numpy as np # linear algebra
import matplotlib
#matplotlib.use('Agg')

import pathfinder
import numpy as np

import utils_lung
import matplotlib.pyplot as plt
import data_transforms
from matplotlib.patches import Circle


def world_to_voxel_coordinates(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

def transform_nodules_vox_coos(nodules, origin, spacing):
    coos_nodules = []
    diameters_original = []
    for nodule in nodules:
        coordinates = world_to_voxel_coordinates(nodule[0:3], origin, spacing)
        coos_nodules.append(coordinates)
        diameters_original.append(nodule[3])
    return coos_nodules, diameters_original



d_labels =  utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

not_found_list = [
"1.3.6.1.4.1.14519.5.2.1.6279.6001.282512043257574309474415322775",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.144883090372691745980459537053",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.801945620899034889998809817499",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.964952370561266624992539111877",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.943403138251347598519939390311",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.123697637451437522065941162930",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.127965161564033605177803085629",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.312127933722985204808706697221",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.177252583002664900748714851615",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.219349715895470349269596532320",

]

patient_id =not_found_list[3]

img, origin, pixel_spacing = utils_lung.read_mhd('/home/frederic/kaggle-dsb3/data/luna/dataset/'+str(patient_id)+".mhd")

cancer_nodules = d_labels[patient_id]

p_transform = {'patch_size': (416, 416, 416),
               'mm_patch_size': (416, 416, 416),
               'pixel_spacing': (1,1,1)
               }

data_out, annotatations_out, tf_total = data_transforms.transform_scan3d(data=img,
                                                                    pixel_spacing=pixel_spacing,
                                                                    p_transform=p_transform,
                                                                    luna_annotations=cancer_nodules,
                                                                    p_transform_augment=None,
                                                                    luna_origin=origin)


print()
#fig = plt.figure()
for i in range(len(annotatations_out)):
    index = int(annotatations_out[i][0])
    print(index)
    fig, ax = plt.subplots(1)
    ax.imshow(data_out[index,:,:], cmap=plt.cm.gray)
    circ = Circle((annotatations_out[i][2],annotatations_out[i][1]),annotatations_out[i][3]/2.0,fill=False,edgecolor="red")
    ax.add_patch(circ)


plt.show()