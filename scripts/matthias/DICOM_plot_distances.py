import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
from utils import paths


from glob import glob

class ImagesNotAlignedError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "ImagesNotAlignedError: To perform the desired operation, given images must be parallel to each other"


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def get_dcm_orientation(dcm_img):

    return [float(element) for element in dcm_img[0x0020, 0x0037].value]


def compute_normal_vector_dcm(dcm_img):

    dcm_orientation=get_dcm_orientation(dcm_img)

    return compute_normal_vector(dcm_orientation[:3],dcm_orientation[-3:])


def compute_normal_vector(vec1,vec2):

    normal_vector=np.cross(vec1,vec2)
    normal_vector=normal_vector/np.sqrt(np.dot(normal_vector,normal_vector))

    return normal_vector



def check_aligned_image_planes(dicom_img1, dicom_img2):

    img_orientation_1 =[float(element) for element in dicom_img1[0x0020, 0x0037].value]
    img_orientation_2 =[float(element) for element in dicom_img2[0x0020, 0x0037].value]
    normal_vector_1=compute_normal_vector(img_orientation_1[:3] , img_orientation_1[-3:])
    normal_vector_2=compute_normal_vector(img_orientation_2[:3] , img_orientation_2[-3:])

    return isclose(np.dot(normal_vector_1,normal_vector_2),1.0)

def compute_distance_between_images(dcm_img1,dcm_img2):

    # first check if the images are aligned
    if check_aligned_image_planes(dcm_img1, dcm_img2) == False:
        raise ImagesNotAlignedError

    origin1 = [float(element) for element in dcm_img1[0x0020, 0x0032].value]
    origin2 = [float(element) for element in dcm_img2[0x0020, 0x0032].value]

    normal_vector2 = compute_normal_vector_dcm(dcm_img2)
    D = -np.dot(normal_vector2, origin2)

    distance = np.abs(np.dot(normal_vector2, origin1) + D) / np.sqrt(
        np.dot(normal_vector2, normal_vector2))

    return distance



if __name__ == '__main__':

    data_dir = paths.DATA_PATH

    patient_folders=os.listdir(data_dir)
    #image_folders=[[working_path + p + '/'+s for s in os.listdir(working_path + p)] for p in patient_folders]

    metadata_dict={}

    distances = []
    unaligned_occurences = []

    for patient_folder in patient_folders:
        patient_path = data_dir+'/' + patient_folder + '/'
        patient_files=glob(patient_path+'/*.dcm')

        patient_masks = []
        patient_images = []

        out_images=[]
        out_masks=[]

        # FIXXXME name of the patient is identical with his folder name, hacky, but ok for now
        patient_id=patient_folder

        image_positions=[]
        image_orientations=[]


        nvectors=[]
        patient_dict={}






        #Try to load all the patient images, this will probably
        dicom_images=[]

        for patient_file in patient_files:
            dicom_images.append(dicom.read_file(patient_file))

        #We are  allowed to do that because I already checked in a previous script that all images are aligned
        dicom_images.sort(key=lambda x: int(x.ImagePositionPatient[2]))



        current_dcm=dicom_images[0]
        image_idx=1
        patient_distances = []
        while(image_idx < len(patient_files)):

            next_dcm=dicom_images[image_idx]



            unaligned_counter=0
            # check if the image plains are aligned
            try:
                actual_distance=compute_distance_between_images(current_dcm, next_dcm)
                #naive_distance = float(next_dcm.SliceLocation) - float(current_dcm.SliceLocation)
                patient_distances.append(actual_distance)

            except ImagesNotAlignedError as not_aligned_error:
                print "Computing image distance: {}".format(not_aligned_error)


            current_dcm=next_dcm
            image_idx=image_idx+1

                # for aligned images in the xy-plane (the optimial case) these should be the same as subtracting the slice properties
        distances.append(patient_distances)
        unaligned_occurences.append(unaligned_counter)

    mean_distances=[np.mean(patient_dist_lst) for patient_dist_lst in distances]
    std_distances=[np.std(patient_dist_lst) for patient_dist_lst in distances]
    max_distances=[np.max(patient_dist_lst) for patient_dist_lst in distances]
    min_distances=[np.min(patient_dist_lst) for patient_dist_lst in distances]
    median_distances=[np.median(patient_dist_lst) for patient_dist_lst in distances]

    fig = plt.figure()


    plt.subplot(221)
    plt.scatter(mean_distances,max_distances)
    plt.xlabel('Average Slice Distance')
    plt.ylabel('Maximum Slice Distance')

    plt.subplot(222)

    plt.scatter(mean_distances,min_distances)
    plt.xlabel('Average Slice Distance')
    plt.ylabel('Minimum Slice Distance')

    plt.subplot(223)

    plt.scatter(mean_distances, median_distances)
    plt.xlabel('Average Slice Distance')
    plt.ylabel('Median Slice Distance')


    ax=plt.subplot(224)

    plt.bar(range(len(mean_distances)), mean_distances,alpha=0.4, color='k',yerr=std_distances)
    #plt.hist(range(len(mean_distances)), mean_distances)
    plt.xlabel('Patient')
    plt.ylabel('Average Slice Distance')
    ax.set_xticks(np.arange(len(mean_distances)))# + width / 2.)
    ax.set_xticklabels([str(element) for element in range(len(mean_distances))])

    plt.savefig('./slice_distances.png')

    plt.show()
