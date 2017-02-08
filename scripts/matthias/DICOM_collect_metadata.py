import dicom
import os


from glob import glob
from utils import paths
import pickle


datadir = "/home/matthias/Documents/playground/dsb3/DSB3Tutorial/stage1_subset/"



def collect_metadata(datadir):

    patient_folders=os.listdir(datadir)

    metadata_dict={}


    for patient_folder in patient_folders:
        patient_path = datadir +'/'+ patient_folder + '/'
        patient_files=glob(patient_path+'/*.dcm')


        # FIXXXME name of the patient is identical with his folder name, hacky, but ok for now
        patient_id=patient_folder

        patient_dict={}
        for patient_file in patient_files:

            dicom_file = dicom.read_file(patient_file)

            image_dict={'position':[float(element) for element in dicom_file[0x0020,0x0032].value],'orientation':[float(element) for element in dicom_file[0x0020,0x0037].value],'rows':float(dicom_file[0x0028,0x0010].value),'cols':float(dicom_file[0x0028,0x0011].value),'frame_of_reference':dicom_file[0x0020,0x0052].value}

            patient_dict[int(dicom_file.InstanceNumber)]=image_dict

        metadata_dict[patient_id]=patient_dict
    return metadata_dict



if __name__ == '__main__':

    data_dir = paths.DATA_PATH

    metadata_dict= collect_metadata(data_dir)

    with file("./image_metadata.npy",mode='w') as pickle_file:
        pickle.dump(metadata_dict,pickle_file)


