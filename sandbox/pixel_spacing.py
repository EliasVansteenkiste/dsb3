import cPickle
import utils_lung
import numpy
import utils
import pathfinder
print("")
zero_list_t = []
one_list_t = []

first = True
i=0
test_data = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_size6_s5_p8a1-20170324-231152/dsb_af4_size6_s5_p8a1-20170324-231152-test.pkl","rb"))

for line in open("/home/frederic/kaggle-dsb3/data/test_labels.csv","r"):
    parts = line.strip().split(";")
    if first:
        first = False
    else:
        # if parts[0][0:5]=="f4d23":
        #     one_list.append(1)
        # else:

        if parts[1]=="1":
            image, pixel_spacing = utils_lung.read_dicom_scan("/home/frederic/kaggle-dsb3/data/stage1/" + parts[0])
            print parts[0], pixel_spacing, test_data[parts[0]]
            one_list_t.append(pixel_spacing)
        i+=1

means = (numpy.mean([x[0] for x in one_list_t]),
                      numpy.mean([x[1] for x in one_list_t]),
                      numpy.mean([x[2] for x in one_list_t]),
                      )

print("Mean: "+str(means))


# train_valid_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
# train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']
#
# valid_data = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_size6_s5_p8a1-20170324-231152/dsb_af4_size6_s5_p8a1-20170324-231152-valid.pkl","rb"))
#
#
# i=0
#
# for line in open("/home/frederic/kaggle-dsb3/data/stage1_labels.csv","r"):
#     parts = line.strip().split(",")
#     if first:
#         first = False
#     else:
#         # if parts[0][0:5]=="f4d23":
#         #     one_list.append(1)
#         # else:screen
#         if parts[0] in valid_pids:
#             if parts[1]=="1":
#                 image, pixel_spacing = utils_lung.read_dicom_scan("/home/frederic/kaggle-dsb3/data/stage1/" + parts[0])
#                 print parts[0],pixel_spacing, valid_data[parts[0]]
#                 one_list_t.append(pixel_spacing)
#             i+=1
#
#     means = (numpy.mean([x[0] for x in one_list_t]),
#                           numpy.mean([x[1] for x in one_list_t]),
#                           numpy.mean([x[2] for x in one_list_t]),
#                           )
#
# print("Mean: "+str(means))