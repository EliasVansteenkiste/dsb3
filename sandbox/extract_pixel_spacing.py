
import os
import sys
sys.path.append("../")
import utils_lung
import cPickle

id2positive_annotations = utils_lung.read_luna_properties("/home/frederic/kaggle-dsb3/data/luna/annotations_extended_mixed.csv")
id2negative_annotations = utils_lung.read_luna_negative_candidates("/home/frederic/kaggle-dsb3/data/luna/candidates_V2.csv")


allids = list(set(id2negative_annotations.keys()+id2positive_annotations.keys()))

info = {}
for pid in allids:
    print(pid)
    patient_path = os.path.join("/home/frederic/kaggle-dsb3/data/luna/dataset_pkl/",pid+".pkl")
    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path)
    info[pid]={"origin":origin,"pixel_spacing":pixel_spacing}

with open( os.path.join("/home/frederic/kaggle-dsb3/data/luna/","origin_pixelspacing.pkl"),"wb") as f:
    cPickle.dump(info,f)