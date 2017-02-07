import numpy as np
import dicom
import os

from glob import glob
from utils import paths


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


working_path = paths.DATA_PATH
patient_folders=os.listdir(working_path)






for patient_folder in patient_folders:

    patient_path = working_path +'/'+ patient_folder + '/'
    patient_files=glob(patient_path+'/*.dcm')

    # FIXXXME name of the patient is identical with his folder name, hacky, but ok for now
    patient_id=patient_folder

    unaligned_occurences=[]

    dicom_images=[]

    for patient_file in patient_files:
        dicom_images.append(dicom.read_file(patient_file))

    # sort by instance number for now (could be wrong of course but sorting for x does not make sense either if we do not know if there are unaligned slices)
    dicom_images.sort(key=lambda x: int(x.InstanceNumber))


    current_dcm=dicom_images[0]
    image_idx=1

    unaligned_counter=0
    while(image_idx < len(patient_files)):

        next_dcm=dicom_images[image_idx]

        patient_distances=[]
        unaligned_counter=0
        # check if the image plains are aligned
        if check_aligned_image_planes(current_dcm, next_dcm)==False:

            print "Warning: Unaligned Slices found!"
            unaligned_occurences.append(next_dcm.InstanceNumber)

        current_dcm=next_dcm
        image_idx=image_idx+1

    if len(unaligned_occurences)>0:
        print "Found unaligned slices {} ({} total) for patient {}".format(unaligned_occurences,len(unaligned_occurences),patient_id)

    else:
        print "No unaligned slices found for patient {}".format(patient_id)

