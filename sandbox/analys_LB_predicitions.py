import cPickle
import numpy as np



train_valid_ids = cPickle.load(open("/home/frederic/kaggle-dsb3/data/dsb_validation_split.pkl","rb"))
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']

valid_loss = []
test_loss = []

test_data_2 = cPickle.load(open("/home/frederic/kaggle-dsb3/metadata/model-predictions/dsb_af25lme_mal2_s5_p8a1-20170330-234414-test.pkl","rb"))


i=1
for line in open("/home/frederic/kaggle-dsb3/metadata/analysis/frederic/LB_labels","r").readlines():
    parts = line.strip().split(" ")
    name = parts[4].strip()[3:-2]

    if name in test_pids:
        test_loss.append(float(parts[1]))

        if float(parts[3][:-1])==1:
            loss = -np.log(test_data_2[name])
        else:
            loss = -np.log(1-test_data_2[name])

        print(str(parts[3][:-1])+","+str(parts[1])+","+str(loss)+","+name)

        #print(line)
    elif name in valid_pids:

        valid_loss.append(float(parts[1]))
    else:
        print("Error")
    i+=1

print("Final valid loss: "+str(np.mean(valid_loss)))
print("Final test loss: "+str(np.mean(test_loss)))