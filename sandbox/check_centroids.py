import os
import cPickle
import pathfinder
import utils_lung

dumpdir = "/home/frederic/kaggle-dsb3/data/luna/nodule_annotations"

anno = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)


for f_name in os.listdir(dumpdir):

    pid = f_name[:-4]
    patient = cPickle.load(open(os.path.join(dumpdir,f_name),"rb"))

    if pid in anno:
        t = anno[pid]
        print()