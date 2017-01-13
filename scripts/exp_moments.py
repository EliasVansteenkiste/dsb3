import numpy as np
import matplotlib.pyplot as plt

# todo: find monotone transformation which transforms data such that it makes the data look like a normal distribution/uniform distribution,
# todo: given estimates of the first k moments

X = np.random.beta(2,10,size=10000)
plt.hist(X, 50, normed=1, facecolor='green', alpha=0.75)

def moment(x,k):
    return np.mean((x-np.mean(x))**k)
# step 1: estimate first k moments
K = 10
moments=[moment(X,k) for k in xrange(K)]
print moments

# 1=1
# 0=0
#   y=x-u
# 1=(X-u)^2
#   y=(x-u)/s
# 0=(X-u)^3
#   y=(x-u)/s
# 3=(X-u)^4

X2 = (X-np.mean(X))/np.sqrt(moment(X,2))
print [moment(X2,k) for k in xrange(K)]
print np.std(X2)

plt.hist(X2, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()

