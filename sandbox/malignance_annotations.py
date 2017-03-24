import xml.etree.ElementTree as ET
import cPickle
import os
rootdir = "/home/frederic/kaggle-dsb3/data/luna_xml"
dumpdir = "/home/frederic/kaggle-dsb3/data/luna/nodule_annotations"

ids = {}
id_list = []
for rootd, subdirs, files in os.walk(rootdir):

    print()

    for file in files:
        tree = ET.parse(os.path.join(rootd,file))
        root = tree.getroot()
        rh=root.findall("{http://www.nih.gov}ResponseHeader")
        if len(rh) > 0:
            id=rh[0].findall("{http://www.nih.gov}SeriesInstanceUid")
            if len(id) > 0:

                id=id[0].text.strip()

                sessions = []
                for reading_session in root.findall("{http://www.nih.gov}readingSession"):

                    reading_session_list =[]
                    for unblinded_nodule in reading_session.findall("{http://www.nih.gov}unblindedReadNodule"):

                        characteristics = unblinded_nodule.findall("{http://www.nih.gov}characteristics")

                        if len(characteristics)>0:
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
                                    y = float(edge.find("{http://www.nih.gov}xCoord").text)
                                    xyz.append((x,y,z))


                            nodule = {}
                            nodule["characteristics"] = char_dict
                            nodule["rois"]=xyz

                            reading_session_list.append(nodule)
                    sessions.append(reading_session_list)

                with open(os.path.join(dumpdir,str(id)+".pkl"),"wb") as f:
                    cPickle.dump(sessions,f)

                # if id in id_list:
                #     print()
                #
                id_list.append(id)
                #ids[id]=sessions


# test = ids["1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824"]
print(len(id_list))
print(len(set(id_list)))