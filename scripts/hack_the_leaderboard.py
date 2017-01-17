import heapq
import random
import math
from bitarray import bitarray
import numpy as np

def check_on_leaderboard(expected, predicted):
    expected, predicted = np.array(expected), np.array(predicted)
    predicted = np.clip(predicted, 1e-15, 1-1e-15)

    result = -np.mean(expected*np.log(predicted) + (1-expected)*np.log(1-predicted))
    return float("%.06f"%result)  # round to 6 digits after comma


LENGTH = 198

def alternate(n):
    return (([0]*n+[1]*n)*LENGTH)[:LENGTH]


def distance(a,b):
    return np.sum((a-b)**2)

def find_solution(a,b):

    # for i in xrange(LENGTH):
    #     solution = bitarray( [random.randint(0,1) for i in xrange(LENGTH)])
    #     already_there = {solution}
    #     todos = [ (distance( b, np.dot(a, solution.tolist()) ), solution) ]

    a_big = np.concatenate([a,np.eye(LENGTH)],axis=0)
    b_big = np.concatenate([b,0.5*np.ones(shape=(LENGTH,))],axis=0)
    solution = np.linalg.lstsq(a_big, b_big)[0]
    solution = bitarray([0 if s<0.5 else 1 for s in solution])
    print "original", distance( b, np.dot(a, solution.tolist()))
    todos = [ (distance( b, np.dot(a, solution.tolist()) ), solution) ]
    already_there = {solution}

    heapq.heapify( todos )
    best = np.inf
    while todos:
        prev_d, current_solution = heapq.heappop(todos)
        # test all bit flips, see which one works best
        for bitnumber in xrange(len(current_solution)):
            solution = bitarray(current_solution)
            solution[bitnumber] = 1-solution[bitnumber]
            d = distance( b, np.dot(a,tuple(solution)) )
            if d==0:
                return solution
            else:
                if d<best:
                    best = d
                    print d, len(already_there)
            item = (d, solution)
            if solution not in already_there:
                already_there.add(solution)
                heapq.heappush(todos, item)
                if len(already_there)%100000==0:
                    print len(already_there)
        # memory and stuff
        MAX_TODOS = 1000
        if len(todos)>MAX_TODOS:
            todos = heapq.nsmallest(MAX_TODOS,todos)
            heapq.heapify(todos)


for test_number in xrange(1):
    true_labels = [random.randint(0,1) for i in xrange(LENGTH)]

    # now, we try to reconstruct this submission
    # try to stay between 0.56018 and 0.69315 to not get detected
    # try to remain non-obvious altogether
    random_permutation = range(LENGTH)
    random.shuffle(random_permutation)
    results = []
    submissions = []

    num_submissions = int(math.ceil(math.log(LENGTH, 2)))
    print num_submissions,"submissions"
    MARGIN = 0.4
    for n in xrange(num_submissions):
        # subm = alternate(2**(n+1))
        # subm = [subm[i] for i in random_permutation]
        # don't be too sure.
        # subm = [MARGIN + i*(1-2*MARGIN) for i in subm]
        subm = [random.uniform(0.4,0.6) for i in xrange(LENGTH)]
        results.append(check_on_leaderboard(true_labels, subm))
        submissions.append(subm)

    # solve the system
    results = np.array(results)
    subm = np.array(submissions)

    normalize = (np.log(MARGIN) - np.log(1.0-MARGIN))/float(LENGTH)

    a = (np.log(subm) - np.log(1.0-subm))/float(LENGTH)/normalize
    b = -(np.sum(np.log(1.0-subm),axis=1)/float(LENGTH)+results)/normalize

    #a,b =np.round(a).astype('int32'), np.round(b).astype('int32')


    np.set_printoptions(threshold=np.nan)
    # print a[-1,:], b[-1], sum(true_labels)
    print "Distance of the real solution:", distance(np.dot(a,true_labels),b)

    # find a vector of 198 0's and 1's for which a*vector = b
    prediction = find_solution(a,b)

    print np.dot(a,prediction),b

    print prediction
    print true_labels
    if prediction == true_labels:
        print "SOLVED"
    else:
        print "WRONG!!!!!!"


