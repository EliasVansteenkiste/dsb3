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

last_chunk = -1
training_errors = []
validation_errors = []
training_idcs = []
validation_idcs=[]

for line in file:
	if 'Chunk' in line :
		last_chunk = int(line.split()[1].split('/')[0])
	if 'Validation loss' in line:
		validation_errors.append(float(line.split(':')[1].rsplit()[0]))
		validation_idcs.append(last_chunk)
	if 'Mean train loss' in line:
		training_errors.append(float(line.split(':')[1].rsplit()[0]))
		training_idcs.append(last_chunk)


print 'training errors'
print training_errors
print training_idcs
print 'validation errors'
print validation_errors
print validation_idcs

plt.plot(training_errors, label='training errors')
plt.plot(validation_errors, label='validation errors')
plt.legend(loc="upper right")
plt.title(sys.argv[1])
plt.xlabel('Epoch')
plt.ylim(0, 0.7)
plt.ylabel('Error')	
plt.savefig(sys.argv[2])


