import lasagne
import theano.tensor as T
from interfaces.objectives import TargetVarDictObjective, MAXIMIZE
from theano_utils import theano_printer

import theano
import theano.tensor as T
try:
    from theano.tensor.extra_ops import searchsorted
except:
    raise "FUCKING UPDATE YOUR THEANO TO VERSION ZERO POINT FUCKING NINE"
import numpy as np
import lasagne
from lasagne.layers import get_output



#================ Turd objective ==================

MAX_INT = np.iinfo(np.int32).max
MAX_FLOAT = np.finfo(np.float32).max


class InterpolatedAucObjective(TargetVarDictObjective):

    optimize = MAXIMIZE

    def __init__(self, input_layers, target_name, delta_auc_instead = False, *args, **kwargs):
        super(InterpolatedAucObjective, self).__init__(input_layers, *args, **kwargs)
        self.target_key = target_name + ":class"
        self.target_vars[self.target_key]  = T.lvector("target_label")
        self.predicted_rank = input_layers["predicted_rank"]
        """
        These must be kept sorted
        """
        self.T = theano.shared(value=np.array([-MAX_FLOAT / 2., MAX_FLOAT / 2.], dtype='float32'), name='T')
        self.labels = theano.shared(value=np.array([1, 0], dtype='int64'), name='labels')
        self.TPR = theano.shared(value=np.array([1, 0], dtype='int64'), name='TPR')
        self.FPR = theano.shared(value=np.array([1, 1], dtype='int64'), name='FPR')
        self.idx_left = theano.shared(value=np.array([0, 0], dtype='int64'), name='idx_left')
        self.idx_right = theano.shared(value=np.array([1, 1], dtype='int64'), name='idx_right')
        self.extrapolate_left = theano.shared(value=np.array([0, 0], dtype='int64'), name='extrapolate_left')
        self.extrapolate_right = theano.shared(value=np.array([0, 0], dtype='int64'), name='extrapolate_right')
        self.time_added = np.array([MAX_INT, MAX_INT], dtype='int32')
        """
        These contain some global information on the previous lists
        """
        self.AUC0 = theano.shared(value=np.int64(0.), name='AUC0')
        self.N_P = theano.shared(value=np.int64(0.), name='N_P')
        self.N_F = theano.shared(value=np.int64(0.), name='N_F')
        """
        These contain some information for maintaining the lists
        """
        self.N_added = 0
        self.delta_auc_instead = delta_auc_instead

    def auc_error(self, T_prediction, true_label):

        #if theano.config.optimizer != "fast_run":
        #    raise Exception("This cannot be differentiated without optimization on")

        # get TPR1, TPR2, label_1, FPR1, T1, T2 from the estimated distribution
        idx = searchsorted(self.T, T_prediction, side='left')

        idx = T.switch(T.eq(idx,self.T.shape[0]),self.T.shape[0]-1,idx)

        # sometimes you need to extrapolate, sometimes you don't. Choose appropriate points
        idx_l = T.switch(T.eq((1-true_label)*self.extrapolate_left [idx-1],1), 0,                   self.idx_left [idx-1]                  )
        idx_r = T.switch(T.eq(   true_label *self.extrapolate_right[idx]  ,1), self.N_F+self.N_P+1, self.idx_right[idx  ], )

        T1 = self.T[idx_l]
        TPR1 = self.TPR[idx_l]
        FPR1 = self.FPR[idx_l]

        T2 = self.T[idx_r]
        TPR2 = self.TPR[idx_r]
        FPR2 = self.FPR[idx_r]

        l = true_label
        f = 1-l

        # go from integers to floats
        AUC1 = ( self.AUC0 + f*TPR1 + l*(self.N_F-FPR1) ).astype('float32')
        dAUC = ( (TPR2 - TPR1)*f - (FPR2 - FPR1 - 1)*l ).astype('float32')
        norm = ( (self.N_P+l) * (self.N_F+f) ).astype('float32')

        # deal with the fact that T_prediction, T1 and T2 can all be equal, and make the gradient behave nicely
        # when that happens.
        # Epsilon is a bad solution, it messes up the gradient!
        coef = T.switch( T.eq(T2,T1), 0.5*(T_prediction-T1), (T_prediction-T1)/(T2-T1) )
        #coef = (T_prediction-T1)/(T2-T1)


        if self.delta_auc_instead:
            return T.switch(T.eq(norm,0.0), 1.0, coef*dAUC/norm)
        AUCt = AUC1 + coef*dAUC
        return T.switch(T.eq(norm,0.0), 2.0, AUCt/norm)


    def add_points(self, predicted, label, never_remove=False):
        idx = np.searchsorted(self.T.get_value(), predicted)

        labels = np.insert(self.labels.get_value(), idx, label,)
        Ts = np.insert(self.T.get_value(), idx, predicted,)

        if never_remove:
            self.time_added = np.insert(self.time_added, idx, MAX_INT)
        else:
            self.time_added = np.insert(self.time_added, idx, self.N_added)
            if isinstance(predicted, list):
                self.N_added += len(predicted)
            else:
                self.N_added += 1

        self._update(labels, Ts)


    def _update(self, labels, Ts):
        self.T.set_value(Ts)
        self.labels.set_value(labels)
        tpr = np.cumsum(labels[::-1])[::-1]
        tpr[0] -= 1
        self.TPR.set_value(tpr)
        fpr = range(1,len(labels)+1)[::-1] - np.cumsum(labels[::-1])[::-1] - 1
        self.FPR.set_value(fpr)

        ll = list(labels)
        idx_left  = [i-ll[i::-1].index(1) for i in xrange(len(ll))]
        idx_right = [i+ll[i:]   .index(0) for i in xrange(len(ll))]
        self.extrapolate_left .set_value( [(l==0)         for l in idx_left ] )
        self.extrapolate_right.set_value( [(r==len(ll)-1) for r in idx_right] )

        if max(idx_left)!=0:
            m = min(i for i in idx_left if i > 0)
            idx_left = [i if i!=0 else m for i in idx_left]
        self.idx_left.set_value(idx_left)

        if min(idx_right)!=len(ll)-1:
            m = max(i for i in idx_right if i < len(ll)-1)
            idx_right = [i if i!=len(ll)-1 else m for i in idx_right]
        self.idx_right.set_value(idx_right)

        l = labels[1:-1:]
        self.AUC0.set_value(np.sum((1-l) * np.cumsum(l[::-1])[::-1]))
        self.N_P.set_value(np.sum(labels)  -1)
        self.N_F.set_value(np.sum(1-labels)-1)


    def remove_points_older_than(self, time):
        keep_idxs = [i for i in xrange(len(self.time_added)) if self.time_added[i] >= (self.N_added - time)]
        labels = self.labels.get_value()[keep_idxs]
        Ts = self.T.get_value()[keep_idxs]
        self.time_added = self.time_added[keep_idxs]
        self._update(labels, Ts)

    def remove_all_points(self, *args, **kwargs):
        self.remove_points_older_than(0)

    @property
    def current_auc(self):
        if 0==(self.N_P.get_value() * self.N_F.get_value()):
            return 1.0
        return 1.0 * self.AUC0.get_value() / (self.N_P.get_value() * self.N_F.get_value())

    def print_status(self):
        print "T:  \t", self.T.get_value()
        print "lbl:\t", self.labels.get_value()
        print "TPR:\t", self.TPR.get_value()
        print "FPR:\t", self.FPR.get_value()
        print "AUC0:\t", self.AUC0.get_value()
        print "N_P:\t", self.N_P.get_value()
        print "N_F:\t", self.N_F.get_value()
        print "l_idx:\t", self.idx_left.get_value()
        print "r_idx:\t", self.idx_right.get_value()
        print "l_ext:\t", self.extrapolate_left.get_value()
        print "r_ext:\t", self.extrapolate_right.get_value()
        print "time:\t", self.time_added
        print "AUC:\t", self.current_auc


    def score_lists(self, predicted, expected):
        from sklearn.metrics import roc_auc_score as auc
        if np.min(expected)==np.max(expected):
            return 1.0
        return auc(expected, predicted)

    def get_loss(self, *args, **kwargs):
        network_predictions = lasagne.layers.helper.get_output(self.predicted_rank, *args, **kwargs)
        target_values = self.target_vars[self.target_key]
        return self.auc_error(network_predictions, target_values)



if __name__=="__main__":
    aucd = InterpolatedAucObjective()
    aucd.add_points(-3,0)
    aucd.add_points(-2,0)
    aucd.add_points(-1,1)
    aucd.add_points( 1,0)
    aucd.add_points( 2,1)
    aucd.add_points( 3,1)
    aucd.remove_all_points()
    aucd.add_points([-3,-2,0,0,2,3],[0,0,1,0,1,1])
    aucd.print_status()

    w, x, y = T.scalar('w'), T.scalar('x'), T.scalar('y')
    z1 = aucd(w*x, y)
    f = theano.function([w, x, y], [z1], on_unused_input='ignore')
    X=0
    W=1
    print f(W,X,0)
    print f(W,X,1)

    gr1 = T.grad(z1, x)
    g = theano.function([w, x, y], [gr1], on_unused_input='ignore')
    print g(W,X,0)
    print g(W,X,1)
