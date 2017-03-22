import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import cPickle as pickle
import time
import sys
from utils.configuration import path_to_importable_string
from utils.paths import MODEL_PATH
import pprint
pp = pprint.PrettyPrinter(indent=4)

print "Looking for the metadata files..."

path = MODEL_PATH + path_to_importable_string(sys.argv[1]) + ".pkl"
print path
files = sorted(glob.glob(path))
print "Plotting..."

NUM_TRAIN_PATIENTS = 417
plt.ion()
for file in files:
#    try:
        filename = os.path.basename(os.path.normpath(file))
        data = pickle.load(open(file, "r"))
        train_losses = data['losses']['training']['objective']
        valid_losses = data['losses']['validation']["validation set"]['objective']
        pp.pprint(data['losses']['validation'])
        fig = plt.figure()

        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
#        mngr.window.setGeometry(50, 100, 640, 545)
        plt.title(filename)
        x_train = np.arange(len(train_losses))+1

        plt.gca().set_yscale('log')
        plt.plot(x_train, train_losses)
        if len(valid_losses)>=1:
            x_valid = np.arange(0,len(train_losses),1.0*len(train_losses)/len(valid_losses))+1
            plt.plot(x_valid, valid_losses)

        if file == files[-1]:
            plt.ioff()
        plt.xlabel("chunks")
        plt.ylabel("error")
        plt.title(filename)
        print filename
        print "min valid loss:", min(valid_losses)
        print "end valid loss:", valid_losses[-1]
plt.show()
#    except:
 #       print "%s is corrupt. Skipping" % file

print "done"