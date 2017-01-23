import heapq
import random
import math
import numpy as np

LENGTH = 4

def check_on_leaderboard(expected, predicted):
    expected, predicted = np.array(expected), np.array(predicted)
    predicted = np.clip(predicted, 1e-15, 1-1e-15)

    result = -np.mean(expected*np.log(predicted) + (1-expected)*np.log(1-predicted))
    return float("%.06f"%result)  # round to 6 digits after comma



def aim_for_value(val):
    return 1./(1.+np.exp(-val*LENGTH))


def int_to_bit_array(val, num_bits):
    res = [1 if c=='1' else 0 for c in bin(val)[2:]]
    if len(res)<num_bits:
        res = [0]*(num_bits-len(res)) + res
    else:
        res = res[-num_bits:]
    return res[::-1]


for test_number in xrange(1):
    true_labels = [random.randint(0,1) for i in xrange(LENGTH)]

    # now, we try to reconstruct this submission
    # try to stay between 0.56018 and 0.69315 to not get detected
    # try to remain non-obvious altogether
    random_permutation = range(LENGTH)
    random.shuffle(random_permutation)
    results = []
    submissions = []

    MARGIN = 0.4

    start = 1e-3
    num_bits = 9

    num_submissions = int(math.ceil(1.0*LENGTH/num_bits))
    print num_submissions,"submissions"

    for n in xrange(num_submissions):
        # subm = alternate(2**(n+1))
        # subm = [subm[i] for i in random_permutation]
        # don't be too sure.
        # subm = [MARGIN + i*(1-2*MARGIN) for i in subm]
        subm = [0.5]*LENGTH
        for i in xrange(num_bits):
            idx = n*num_bits+i
            if idx<LENGTH:
                subm[idx] = aim_for_value(start * 2**(num_bits-i-1))
        results.append(check_on_leaderboard(true_labels, subm))
        submissions.append(subm)

    # print results
    # solve the system
    results = np.array(results)
    subm = np.array(submissions)

    print np.log(2)
    res = []
    for r in results:
        for i in xrange(num_bits):
            print
            print "current",r
            p = aim_for_value(start * 2**(num_bits-i-1))
            if r>np.log(2):
                res.append(0)
                influence = np.log(1.-p)+np.log(2)
                r += influence
            else:
                res.append(1)
                influence = np.log(p)+np.log(2)
                r -= influence
            print "influence",influence

    print res
    prediction = res[:LENGTH]
    print len(prediction)

    print sum([int_to_bit_array(int(np.log(2)/start), num_bits) for r in results],[])[::-1]
    print prediction
    print true_labels
    if prediction == true_labels:
        print "SOLVED"
    else:
        print "WRONG!!!!!!"


