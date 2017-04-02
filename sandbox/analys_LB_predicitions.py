import cPickle
import numpy as np



train_valid_ids = cPickle.load(open("/home/frederic/kaggle-dsb3/data/dsb_validation_split.pkl","rb"))
train_pids, valid_pids, test_pids = train_valid_ids['training'], train_valid_ids['validation'], train_valid_ids['test']

valid_loss = []
test_loss = []

i=1
for line in open("/home/frederic/kaggle-dsb3/metadata/analysis/frederic/LB_labels","r").readlines():
    parts = line.strip().split(" ")
    name = parts[4].strip()[3:-2]

    if name in test_pids:
        test_loss.append(float(parts[1]))
        print(line)
    elif name in valid_pids:

        valid_loss.append(float(parts[1]))
    else:
        print("Error")
    i+=1

print("Final valid loss: "+str(np.mean(valid_loss)))
print("Final test loss: "+str(np.mean(test_loss)))