import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import cPickle as pickle
import time
import sys

filename = sys.argv[1]
file = open(filename)

training_set_found = False
stupid_underline = False
validation_set_found = False

training_errors = []
validation_errors = []

for line in file:
	if not training_set_found and not validation_set_found:
		if 'training set (' in line:
			training_set_found = True
		elif 'validation set (' in line:
			validation_set_found = True
	elif training_set_found:
		if 'objective' in line:
			training_errors.append(float(line.split(':')[1].rsplit()[0]))
			training_set_found = False
	elif validation_set_found:
		if 'objective' in line:
			validation_errors.append(float(line.split(':')[1].rsplit()[0]))
			validation_set_found = False

print 'training errors'
print training_errors
print 'validation errors'
print validation_errors

plt.plot(training_errors, label='training errors')
plt.plot(validation_errors, label='validation errors')
plt.legend(loc="upper right")
plt.title(sys.argv[1])
plt.xlabel('Epoch')
plt.ylabel('Error')	
plt.savefig(sys.argv[2])


