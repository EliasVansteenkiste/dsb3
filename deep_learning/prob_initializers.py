from lasagne.init import Initializer
from lasagne.utils import floatX

import numpy as np


class NodulePriorSigmoid(Initializer):

    def __init__(self,class_prior,num_nodules):

        self.nodule_prior=np.power(class_prior,1./num_nodules)

    def sample(self,shape):

        return floatX(np.ones(shape) *-np.log((1./self.nodule_prior)-1) )




class NodulePriorRectifier(Initializer):

    def __init__(self,class_prior,num_nodules):

        self.nodule_prior=np.power(class_prior,1./num_nodules)

    def sample(self,shape):

        return floatX(np.ones(shape) * -np.log(self.nodule_prior))



        
