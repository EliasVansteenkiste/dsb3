import utils_lung
import glob
import skimage.measure
import numpy as np

lung_masks_path = "/data/dsb3/luna/seg-lungs-LUNA16/seg-lungs-LUNA16/"

files = glob.glob(lung_masks_path + '/*.mhd')
print files

lung_sizes = []
for f in files:
    lung_mask, _, _ = utils_lung.read_mhd(f)
    lung_mask = np.asarray(lung_mask > 0, dtype='int8')
    label_image = skimage.measure.label(lung_mask)
    region_props = skimage.measure.regionprops(label_image)
    sorted_regions = sorted(region_props, key=lambda x: x.area, reverse=True)
    lung_sizes.append(sorted_regions[0].area)
    print sorted_regions[0].area

print np.min(lung_sizes), np.max(lung_sizes)
