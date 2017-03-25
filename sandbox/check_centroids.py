import os
import cPickle
import pathfinder
import utils_lung
import numpy as np

def read_mhd_file(path):
    # SimpleITK has trouble with multiprocessing :-/
    import SimpleITK as sitk    # sudo pip install --upgrade pip; sudo pip install SimpleITK
    itk_data = sitk.ReadImage(path.encode('utf-8'))
    pixel_data = sitk.GetArrayFromImage(itk_data)
    origin = np.array(list(itk_data.GetOrigin()))
    spacing = np.array(list(itk_data.GetSpacing()))
    return pixel_data, origin, spacing
def load_patient_data(path):
    result = dict()
    pixel_data, origin, spacing = read_mhd_file(path)
    result["pixeldata"] = pixel_data.T  # move from zyx to xyz
    result["origin"] = origin  # move from zyx to xyz
    print origin
    result["spacing"] = spacing  # move from zyx to xyz
    print spacing
    return result
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

def voxel_to_world_coordinates(voxel_coord, origin, spacing):

    stretched_voxel_coord = voxel_coord *spacing
    world_coord = np.absolute(stretched_voxel_coord + origin)
    return world_coord

def L2(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**(0.5)

dumpdir = "/home/frederic/kaggle-dsb3/data/luna/nodule_annotations"

anno = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)


for f_name in os.listdir(dumpdir):

    pid = f_name[:-4]
    patient = cPickle.load(open(os.path.join(dumpdir,f_name),"rb"))

    if pid in anno:

        luna_nodules = anno[pid]
        for doctor in patient:
            for nodule in doctor:
                for ln in luna_nodules:
                    if "centroid_xyz" in nodule:
                        if L2(ln[0:3],nodule["centroid_xyz"][::-1]) < 5:
                            print("Very close")
                        else:
                            print("Not so close")
                    else:
                        print("Error")