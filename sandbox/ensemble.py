import numpy
import cPickle

test_data_1 = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_a_liolme16_c3_s2_p8a1-20170322-213835/dsb_a_liolme16_c3_s2_p8a1-20170322-213835-public_LB.pkl","rb"))
#test_data_1 = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af1_c2_s5_p8a1-20170324-111246/dsb_af1_c2_s5_p8a1-20170324-111246-test.pkl","rb"))

test_data_2 = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_c3_s5_p8a1-20170323-101206/dsb_af4_c3_s5_p8a1-20170323-101206-test.pkl","rb"))
#test_data_2 = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_c3_s2_p8a1-20170319-123808/dsb_af4_c3_s2_p8a1-20170319-123808-test.pkl","rb"))
#test_data_2 = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_size6_s5_p8a1-20170324-231152/dsb_af4_size6_s5_p8a1-20170324-231152-test.pkl","rb"))


print("")
zero_list_t = []
one_list_t = []

func = numpy.max

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
            zero_list_t.append(func([test_data_1[parts[0]],test_data_2[parts[0]]]))
        else:
            one_list_t.append(func([test_data_1[parts[0]],test_data_2[parts[0]]]))

print("Zeros test: "+str(sum(zero_list_t)/float(len(zero_list_t))))
print("Ones test: "+str(sum(one_list_t)/float(len(one_list_t))))

p = 1.05

a = numpy.asarray(one_list_t)
#a = numpy.sign(a)*numpy.abs(a-0.5)**p+0.5

b = numpy.asarray(zero_list_t)
#b = numpy.sign(b)*numpy.abs(b-0.5)**p+0.5

a = numpy.log(a)
b = numpy.log(1-b)
c = numpy.concatenate([a,b])

print("Log loss "+str(-numpy.mean(c)))

valid_data_1 = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_c3_s2_p8a1-20170319-123808/dsb_af4_c3_s2_p8a1-20170319-123808.pkl","rb"))
valid_data_2 = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_c3_s5_p8a1-20170323-101206/dsb_af4_c3_s5_p8a1-20170323-101206.pkl","rb"))


zero_list_v = []
one_list_v = []

first = True
for line in open("/home/frederic/kaggle-dsb3/data/stage1_labels.csv","r"):
    parts = line.strip().split(",")
    if first:
        first = False
    else:
        if parts[0] in valid_data_1:
            if parts[1]=="0":
                zero_list_v.append(func([valid_data_1[parts[0]],valid_data_2[parts[0]]]))
            else:
                one_list_v.append(func([valid_data_1[parts[0]],valid_data_2[parts[0]]]))

print("Zeros valid: "+str(sum(zero_list_v)/float(len(zero_list_v))))
print("Ones valid: "+str(sum(one_list_v)/float(len(one_list_v))))

a = numpy.log(numpy.asarray(one_list_v))
b = numpy.log(1-numpy.asarray(zero_list_v))
c = numpy.concatenate([a,b])

print("Log loss "+str(-numpy.mean(c)))