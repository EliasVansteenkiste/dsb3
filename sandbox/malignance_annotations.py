import xml.etree.ElementTree as ET
import cPickle
import os
from bs4 import BeautifulSoup
import numpy
import numpy as np
import utils_lung

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
    #print origin
    result["spacing"] = spacing  # move from zyx to xyz
    #print spacing
    return result

def voxel_to_world_coordinates(voxel_coord, origin, spacing):

    stretched_voxel_coord = voxel_coord *spacing
    world_coord = stretched_voxel_coord + origin
    return world_coord

def voxel_to_world_coordinates_flipped(voxel_coord, origin, spacing):

    stretched_voxel_coord = voxel_coord *spacing
    world_coord = -1.0*(stretched_voxel_coord - origin)
    return world_coord


rootdir = "/home/frederic/kaggle-dsb3/data/luna_xml"
dumpdir = "/home/frederic/kaggle-dsb3/data/luna/nodule_annotations"
list_luna = "/home/frederic/kaggle-dsb3/data/LUNA - List of included scans.html"

not_found_list = [
"1.3.6.1.4.1.14519.5.2.1.6279.6001.282512043257574309474415322775",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.144883090372691745980459537053",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.801945620899034889998809817499",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.964952370561266624992539111877",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.943403138251347598519939390311",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.123697637451437522065941162930",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.127965161564033605177803085629",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.312127933722985204808706697221",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.177252583002664900748714851615",
"1.3.6.1.4.1.14519.5.2.1.6279.6001.219349715895470349269596532320",

]

found_list = ["1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860"]

#### LUNA ids #####
soup = BeautifulSoup(open(list_luna,"r"), 'html.parser')

rows = soup.find("tbody").find_all("tr")

serie_study = []
for row in rows:
    cells = row.find_all("td")
    if cells[0].get_text().strip()!="PatientID":
        study = cells[1].get_text().strip()
        serie = cells[2].get_text().strip()
        serie_study.append((serie,study))

#### nodule annotations ####
print(len(set(serie_study)))

serie = [x[0] for x in serie_study]

ids = {}
id_list = []
for rootd, subdirs, files in os.walk(rootdir):

    print()

    for file in files:
#rootd = "/home/frederic/kaggle-dsb3/data/luna_xml"
#file = "tcia-lidc-xml/186/273.xml"
        tree = ET.parse(os.path.join(rootd,file))
        root = tree.getroot()
        rh=root.findall("{http://www.nih.gov}ResponseHeader")
        if len(rh) > 0:
            serie_id=rh[0].findall("{http://www.nih.gov}SeriesInstanceUid")
            study_id = rh[0].findall("{http://www.nih.gov}StudyInstanceUID")
            if len(serie_id) > 0 and len(study_id)>0:

                ser_id = serie_id[0].text.strip()
                stu_id = study_id[0].text.strip()
                #print(ser_id)

                if ser_id in serie:

                    if ser_id in not_found_list:

                        sessions = []
                        chars_found = 0
                        for reading_session in root.findall("{http://www.nih.gov}readingSession"):

                            reading_session_list =[]

                            for unblinded_nodule in reading_session.findall("{http://www.nih.gov}unblindedReadNodule"):

                                characteristics = unblinded_nodule.findall("{http://www.nih.gov}characteristics")

                                if len(characteristics)>0:
                                    chars_found+=1
                                    char_dict = {}
                                    for el in characteristics[0]._children:
                                        char_dict[el.tag[20:]]=float(el.text.strip())

                                    roi_dict = {}

                                    rois = unblinded_nodule.findall("{http://www.nih.gov}roi")
                                    xyz = []
                                    for roi in rois:
                                        z = float(roi.find("{http://www.nih.gov}imageZposition").text)


                                        for edge in roi.findall("{http://www.nih.gov}edgeMap"):
                                            x = float(edge.find("{http://www.nih.gov}xCoord").text)
                                            y = float(edge.find("{http://www.nih.gov}yCoord").text)
                                            xyz.append((x,y,z))


                                    nodule = {}
                                    nodule["characteristics"] = char_dict
                                    #nodule["rois"]=xyz
                                    nodule["centroid"]=(numpy.mean([x[0] for x in xyz]),
                                                        numpy.mean([x[1] for x in xyz]),
                                                        numpy.mean([x[2] for x in xyz]),
                                                        )



                                    reading_session_list.append(nodule)
                            sessions.append(reading_session_list)

                        if chars_found>0:
                        #
                        #if ser_id in not_found_list:
                            for s in sessions:
                                for nodule in s:
                                    img, origin, pixel_spacing = utils_lung.read_mhd(
                                        '/home/frederic/kaggle-dsb3/data/luna/dataset/' + str(ser_id) + ".mhd")

                                    if ser_id in not_found_list:
                                        xy_tf = voxel_to_world_coordinates_flipped(nodule["centroid"][0:2][::-1], origin[1:3],
                                                                           pixel_spacing[1:3])
                                    else:
                                        xy_tf = voxel_to_world_coordinates(nodule["centroid"][0:2][::-1], origin[1:3],
                                                                           pixel_spacing[1:3])
                                    nodule["centroid_xyz"] = (xy_tf[1], xy_tf[0], nodule["centroid"][2])
                                    nodule.pop('centroid', None)

                        with open(os.path.join(dumpdir,str(ser_id)+".pkl"),"wb") as f:
                            cPickle.dump(sessions,f)

                        # if ser_id in id_list and ser_id in not_found_list:
                        #     temp=ids[ser_id]
                        #     print()
                        print(ser_id)
                        id_list.append(ser_id)
                        ids[ser_id]=sessions


# test = ids["1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824"]
print(len(id_list))
print(len(set(id_list)))

