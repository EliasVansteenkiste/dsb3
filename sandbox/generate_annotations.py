import os
import cPickle
import pathfinder
import utils_lung
import numpy as np

def L2(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**(0.5)

dumpdir = "/home/frederic/kaggle-dsb3/data/luna/nodule_annotations"

characteristics = ["calcification","internalStructure","lobulation","malignancy","margin","sphericity","spiculation","subtlety","texture"]

anno = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

cancers=0
luna_lines = []
for f_name in os.listdir(dumpdir):

    pid = f_name[:-4]
    patient = cPickle.load(open(os.path.join(dumpdir,f_name),"rb"))

    if pid in anno:

        # if pid!="1.3.6.1.4.1.14519.5.2.1.6279.6001.267957701183569638795986183786":
        #     continue

        luna_nodules = anno[pid]
        found = {}

        for doctor in patient:
            for nodule in doctor:
                distances = []
                for i in range(len(luna_nodules)):
                    if "centroid_xyz" in nodule:
                        distances.append( L2(luna_nodules[i][0:3],nodule["centroid_xyz"][::-1]))
                    else:
                        print("Error")

                min_distance_index = np.argmin(distances)
                if distances[min_distance_index] < 10:
                    if min_distance_index in found:
                        found[min_distance_index].append(nodule)
                    else:
                        found[min_distance_index] = [nodule]
                else:
                    # print("Not so close")
                    pass

        local_counts = 0
        for i,centroids in found.items():
            if len(centroids)>=3:


                # find 3 closest ones to centroid
                distances = []
                for c in centroids:
                    distances.append((L2(luna_nodules[i][0:3], c["centroid_xyz"][::-1]),c))
                d_sorted=sorted(distances, key=lambda x: x[0])

                selection_centroids=[]
                selection_centroids.extend([x[1] for x in d_sorted[0:3]])

                if len(centroids) > 3:
                    if d_sorted[2][0]*20 > d_sorted[3][0]:
                        selection_centroids.append(d_sorted[3][1])
                    else:
                        print("Too big")


                luna_line = str(pid)+","\
                            +str(luna_nodules[i][2])+","\
                            +str(luna_nodules[i][1])+","+\
                            str(luna_nodules[i][0])+","+str(luna_nodules[i][3])+","
                for characteristic in characteristics:
                    value = np.mean([x["characteristics"][characteristic] for x in selection_centroids])
                    luna_line+=str(value)+","

                luna_lines.append(luna_line[:-1])
                cancers+=1



print("Number of different nodules: "+str(cancers))

luna_header = "seriesuid,coordX,coordY,coordZ,diameter_mm,"
luna_header += ",".join(characteristics)

file = open("/home/frederic/kaggle-dsb3/data/luna/annotations_extended_mean.csv","w")
file.write(luna_header+"\n")
file.write("\n".join(luna_lines))
file.close()