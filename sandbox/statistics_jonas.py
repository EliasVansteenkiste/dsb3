import cPickle
import numpy
import matplotlib.pyplot as plt


test_data = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_c3_s2_p8a1-20170319-123808/dsb_af4_c3_s2_p8a1-20170319-123808-test.pkl","rb"))

print("")
zero_list_t = []
one_list_t = []

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
            zero_list_t.append(test_data[parts[0]])
        else:
            one_list_t.append(test_data[parts[0]])

print("Zeros test: "+str(sum(zero_list_t)/float(len(zero_list_t))))
print("Ones test: "+str(sum(one_list_t)/float(len(one_list_t))))

a = numpy.log(numpy.asarray(one_list_t))
b = numpy.log(1-numpy.asarray(zero_list_t))
c = numpy.concatenate([a,b])

print("Log loss "+str(-numpy.mean(c)))


valid_data = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/frederic/dsb_af4_c3_s2_p8a1-20170319-123808/dsb_af4_c3_s2_p8a1-20170319-123808-valid.pkl","r"))

zero_list_v = []
one_list_v = []

first = True
for line in open("/home/frederic/kaggle-dsb3/data/stage1_labels.csv","r"):
    parts = line.strip().split(",")
    if first:
        first = False
    else:
        if parts[0] in valid_data:
            if parts[1]=="0":
                zero_list_v.append(valid_data[parts[0]])
            else:
                one_list_v.append(valid_data[parts[0]])

print("Zeros valid: "+str(sum(zero_list_v)/float(len(zero_list_v))))
print("Ones valid: "+str(sum(one_list_v)/float(len(one_list_v))))

bins = numpy.arange(0,1.01,0.05)

plt.figure()
plt.hist([zero_list_v,zero_list_t],normed=True, bins=bins,color=["blue","red"],stacked=False,label=["valid","test"])
plt.legend(prop={'size': 10})
plt.title("Zero probs")
plt.xlabel('Probability')
plt.ylabel('Counts')
plt.xticks(numpy.arange(0,1.01,0.1))
plt.grid(True)

plt.hist([one_list_v,one_list_t],normed=True, bins=bins,color=["blue","red"],stacked=False,label=["valid","test"])
plt.legend(prop={'size': 10})
plt.title("One probs")
plt.xlabel('Probability')
plt.ylabel('Counts')
plt.xticks(numpy.arange(0,1.01,0.1))
plt.grid(True)

plt.show()

