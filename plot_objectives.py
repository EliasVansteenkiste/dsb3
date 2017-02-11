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

training_errors = []
validation_errors = []

for line in file:
	if 'Validation loss' in line:
		validation_errors.append(float(line.split(':')[1].rsplit()[0]))
	if 'Mean train loss' in line:
		training_errors.append(float(line.split(':')[1].rsplit()[0]))

print 'training errors'
print training_errors
print 'validation errors'
print validation_errors

plt.plot(training_errors, label='training errors')
plt.plot(validation_errors, label='validation errors')
plt.legend(loc="upper right")
plt.title(sys.argv[1])
plt.xlabel('Epoch')
plt.ylim(0, 0.7)
plt.ylabel('Error')	
plt.savefig(sys.argv[2])


