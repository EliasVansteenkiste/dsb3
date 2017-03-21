import cPickle
import numpy

test_data = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_c3_s2_p8a1-20170319-123808/dsb_af4_c3_s2_p8a1-20170319-123808-test.pkl","rb"))

print("")
zero_list = []
one_list = []

first = True
for line in open("/home/frederic/kaggle-dsb3/data/test_labels.csv","r"):
    parts = line.strip().split(";")
    if first:
        first = False
    else:
        # if parts[0][0:5]=="f4d23":
        #     one_list.append(1)
        # else:
        if parts[1]=="0":
            zero_list.append(test_data[parts[0]])
        else:
            one_list.append(test_data[parts[0]])

print("Zeros test: "+str(sum(zero_list)/float(len(zero_list))))
print("Ones test: "+str(sum(one_list)/float(len(one_list))))

a = numpy.log(numpy.asarray(one_list))
b = numpy.log(1-numpy.asarray(zero_list))
c = numpy.concatenate([a,b])

print("Log loss "+str(-numpy.mean(c)))

valid_data = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_c3_s2_p8a1-20170319-123808/dsb_af4_c3_s2_p8a1-20170319-123808-valid.pkl","r"))

zero_list = []
one_list = []

first = True
for line in open("/home/frederic/kaggle-dsb3/data/stage1_labels.csv","r"):
    parts = line.strip().split(",")
    if first:
        first = False
    else:
        if parts[0] in valid_data:
            if parts[1]=="0":
                zero_list.append(valid_data[parts[0]])
            else:
                one_list.append(valid_data[parts[0]])

print("Zeros valid: "+str(sum(zero_list)/float(len(zero_list))))
print("Ones valid: "+str(sum(one_list)/float(len(one_list))))