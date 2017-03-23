import utils_lung
import pathfinder
import numpy as np

id2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

diameters = []
for v in id2annotations.itervalues():
    for vv in v:
        diameters.append(vv[-1])

diameters = np.array(diameters)

print 0.5 * len(np.where(diameters > 30)[0]) / len(diameters)
print 0.5 * len(np.where((diameters < 30) * (diameters > 20))[0]) / len(diameters)
print 0.5 * len(np.where((diameters < 20) * (diameters > 8))[0]) / len(diameters)
print 0.5 * len(np.where((diameters < 8) * (diameters > 4))[0]) / len(diameters)
# print 1. * len(np.where(np.array(diameters) < 4)[0]) / len(diameters)

n = (2038 + 1034 + 268 + 16)

print '-------------------------'
print 2038. / n
print 1034. / n
print 268. / n
print 16. / n
