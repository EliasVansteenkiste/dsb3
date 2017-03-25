import os
import cPickle
import pathfinder
import utils_lung
import numpy as np

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