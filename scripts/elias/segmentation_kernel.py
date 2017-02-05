# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import glob
import random
import csv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
from collections import defaultdict
from utils import paths

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

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 
    plt.axis('scaled')


def get_segmented_lungs(im, z=-1, plot=False, savefig=False):
    '''
    This funtion segments the lungs from a given 2D slice.
    '''


    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    #binary = im < 604
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = -1000
    #im[get_high_vals] = 0
    if plot == True:
        print im.shape
        np.arange(im.shape[0])
        np.arange(im.shape[1])


        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone) 

    if savefig:
        plt.savefig('test_'+str(z)+'.jpg')
        
    return im

def segment_lung_from_ct_scan(slices, plot=False, savefig=False):
    slices
    return np.asarray([get_segmented_lungs(slice, idx, plot=plot, savefig=savefig) for idx, slice in enumerate(slices)])

def patient_name_from_file_name(patient_file):
    return os.path.splitext(os.path.basename(patient_file))[0]

def read_luna_labels(location):
    # step 1: load the file names
    file_list = sorted(glob.glob(location+"*.mhd"))
    # count the number of data points

    # make a stratified validation set
    # note, the seed decides the validation set, but it is deterministic in the names
    random.seed(317070)
    patient_names = [patient_name_from_file_name(f) for f in file_list]

    # load the filenames and put into the right dataset
    labels_as_dict = defaultdict(list)

    with open(paths.LUNA_LABELS_PATH, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader)  # skip the header
        for row in reader:
            label = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
            labels_as_dict[str(row[0])].append(label)
    return labels_as_dict

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

d_labels = read_luna_labels('/local/eavsteen/dsb3/storage/data/dsb3/luna/dataset')


test_patient = load_patient_data('/local/eavsteen/dsb3/storage/data/dsb3/luna/dataset/1.3.6.1.4.1.14519.5.2.1.6279.6001.121391737347333465796214915391.mhd')
print test_patient.keys()
#print d_labels.keys()
cancer_nodules =  d_labels['1.3.6.1.4.1.14519.5.2.1.6279.6001.121391737347333465796214915391']
print 'cancer nodules'
print 'original coos'
print cancer_nodules 
nodules_coos, diameters_original = transform_nodules_vox_coos(cancer_nodules,test_patient['origin'],test_patient['spacing'])
print 'vox coos'
print nodules_coos
print diameters_original


slices = test_patient['pixeldata']
slices = slices.swapaxes(0,2)
print 'histogram', np.histogram(slices)


slices[slices < -2000] = - 2000
plt.imshow(slices[60], cmap=plt.cm.gray)
plt.savefig('test1.jpg')

plot_ct_scan(slices)
plt.savefig('test2.pdf')

get_segmented_lungs(slices[60], 60, True)
plt.savefig('test3.pdf')

segmented_ct_scan = segment_lung_from_ct_scan(slices)
plot_ct_scan(segmented_ct_scan)
plt.savefig('test4.pdf')

segmented_ct_scan[segmented_ct_scan < -400] = -2000
plot_ct_scan(segmented_ct_scan)
plt.savefig('test5.pdf')



#circle2 = plt.Circle((5, 5), 0.5, color='b', fill=False)

