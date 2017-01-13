from application.data import KaggleSFDataLoader
from interfaces.data_loader import INPUT, TRAINING
from utils.memoize import Memoize

preprocessors = []


#####################
#     training      #
#####################
training_data = KaggleSFDataLoader(sets={"training": 1.0},
                                 epochs=1,
                                 preprocessors=preprocessors,
                                 multiprocess=False
                                )

chunk_size = 16

training_data.prepare()
batch = training_data.generate_batch(
    chunk_size=chunk_size,
    required_input={"kaggle-sf:rgb":(chunk_size,3,640,480)},
    required_output={"kaggle-sf:sample_id":None},

)

import matplotlib.pyplot as plt
import numpy as np
import skimage.color
plt.figure()

ids = sorted(training_data.indices[TRAINING])

plot = None
title = None

known = [[1,2,9,10,24,86,137,205,210,341,390,393,396,397,411,416,437,456,473,487,496] # female black hair purple
         ,[3, 13, 19,25,36,38, 65,70,77,89,93,107,124,131,133,134,156,160,181,186,197,247,248,254,262,270,321,335,391,399,419,424,425,442,472,480,495] # glasses green shirt
         , [6,22, 44,55,90,116,121,144,171,198,212,263,278,284,309,325,330,336,344,345,352,388,435,452,461,474,481,490,492,497] # female beany african
         , [7,16,31,80,112,113,129,130,139,143,188,192,224,243,252,255,273,285,320,358,364,368,400,406,426,429,432,433,443,471,484] # glasses green shirt short sleeve
         , [0,28,54,63,67,96,101,167,179,185,220,301,346,370,377,423] # stripe zero
         , [8,11,74,104,108,148,201,237,295,312,327,373,378,379,395,412,440,447,478,479,500] #indian purple shirt
         , [37,39,57,79,84,88,102,162,190,199,214,219,235,260,268,283,288,303,331,348,403,410,422,446,485] # Paul met chauffeur hoed
         , [14,58,61,99,111,126,155,184,203,436,464,469] # fatty
         , [34,41,60,146,147,166,226,333,357,371,392,398] # oma punk hair
         , [4,66,87,125,136,178,187,211,253,261,290,317,372,401,415,441,454,476] # asian horizontal black on white stripes
         , [17,26,68,72,127,161,202,223,350,417] # indian white business shirt
         , [20,59,73, 75,180,225,241,300,362,394,408,470] # indian white/red shirt short sleeve

         , [5,35,43,62,69,123,150,170,242,279,287,296,298,402,418] # old short sleeve blonde blue schubben sierraad
         , [12,47,94,98,114,154,157,158,163,177,200,216,217,221,250,259,269,271,314,339,359,369,384,420,434,444,463,486] # blue asian with red scarf
         , [15,49,106,117,159,165,175,182,222,277,302,375,421,449] # blond oma kaki shirt
         , [18,46,78,100,110,122,206,213,230,246,267,275,299,306,329,343] # asian granny purple dress black shirt
         , [21,50,52,64,115,140,195,240,249,293,318,323,337,428,465,482,489,493] # asian kid blue shirt grey scarf
         , [23,32,103,118,119,228,251,276,291,315,322,338,349,356,448,458,462,499] # Marie sunglasses dotted shirt
         , [27,42,109,151,164,280,289,310,351,355,380,382,387,445,453,455] # Thomas sunglasses wood shirt
         , [56,81,97,153,169,194,239,245,292,294,304,324,340,374,381,404,405,413,438,439,450] # wood shirt average guy
         , [29,51,53,71,141,183,191,232,256,257,311,328,385] # granny grey shirt short small scarf
         , [30,48,82,95,135,172,173,196,209,218,227,238,266,297,307,313,332,354,366,389] # asian red blue shirt white stripe
         , [33,45,105,128,138,142,208,272,274,281,282,353,407,409,460,475] # old blond black business jacket glasses
         , [40,76,83,85,120,149,176,204,233,264,305,360,363,365,383,414,427,430,491] # african black shirt
         , [91,92,132,168,174,189,207,215,229,234,236,265,286,308,316,319,342,347,367,376,386,431,457,467,468,477,483,494] # female blue horizontal stripe white scarf
         , [145,152,193,231,244,258,326,334,361,451,459,466,488,498] # dark indian bright shirt
         ]
ids = [i for i in ids if i not in sum(known,[])]
"""
for hand labeling
for id in ids + sum(known,[]):

    image = training_data.load_sample(id, input_keys_to_do=["kaggle-sf:rgb"], output_keys_to_do=[])[INPUT]["kaggle-sf:rgb"]

    if plot is None:
        plot = plt.imshow(-np.transpose(image,[2,1,0]), interpolation='none')
        title = plt.title(str(id))
    else:
        plot.set_data(-np.transpose(image,[2,1,0]))
        title.set_text(str(id))
    plt.pause(0.1)
    raw_input()
"""



import sklearn.cluster
X = np.array(training_data.indices[TRAINING], dtype='uint32')[:501,None]

#[17  9  6 12 10 14 15  3  1  4 19  1  8 12 11 16  7  2 13 12 18  0 20 21  5]

@Memoize
def get_image(id):
    image = training_data.load_sample(id, input_keys_to_do=["kaggle-sf:rgb"], output_keys_to_do=[])[INPUT]["kaggle-sf:rgb"]
    image = np.transpose(image, [1,2,0]).astype('uint8')
    image = skimage.color.rgb2hsv(image)
    image = image[:,:,0]
    return image

def mymetric(*args):
    ids = [int(a[0]) for a in args]
    images = [get_image(id) for id in ids]
    #dist = np.count_nonzero(np.logical_not(np.isclose(images[0], images[1], atol=0.1))) * np.count_nonzero(np.logical_not(np.isclose(images[0], images[1], atol=0.01)))
    dist = np.sum((np.abs(images[0] - images[1]))**0.1)
    #hist = [np.histogram(im.flatten(), bins=256,density=True,range=(0.0,1.0))[0] for im in images]
    #crps = np.mean((np.cumsum(hist[0]) - np.cumsum(hist[1]))**2)
    print ids, dist
    return dist



def get_class(i):
    for class_id, cl_ids in enumerate(known):
        if i in cl_ids:
            return class_id


distance_matrix = np.zeros((X.shape[0], X.shape[0])) #np.zeros((100,100)) #np.zeros((X.shape[0], X.shape[0]))
inside_class_distance = []
outside_class_distance = []
for i in xrange(distance_matrix.shape[0]):
    for j in xrange(i,distance_matrix.shape[1]):
        dist = mymetric([i],[j])
        distance_matrix[i,j] = dist
        distance_matrix[j,i] = dist
        if i!=j:
            if get_class(i)==get_class(j):
                inside_class_distance.append(dist)
            else:
                outside_class_distance.append(dist)


import cPickle as pickle
pickle.dump(distance_matrix, open('distance_matrix.pkl','w'))

#plt.hist(inside_class_distance, 50, normed=1, facecolor='green', alpha=0.75)
#plt.hist(outside_class_distance, 50, normed=1, facecolor='red', alpha=0.75)

#plt.show()

dbscan = sklearn.cluster.DBSCAN(eps=100000, min_samples=10, metric=mymetric)
labels = dbscan.fit_predict(X)

print labels

import cPickle as pickle
pickle.dump(labels, open('labels.pkl','w'))
