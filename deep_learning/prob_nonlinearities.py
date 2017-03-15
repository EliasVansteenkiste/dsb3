import theano.tensor as T

def prob_nonlinearity(x):

    probs_benign=T.prod(x,axis=1,keepdims=True)
    probs_malignant=1. - probs_benign
    output=T.concatenate([probs_benign,probs_malignant],axis=1)
    
    return output

def logprob_nonlinearity(x):

    neg_log_probs_sum = T.sum(x,axis=1,keepdims=True)
    probs_benign=T.exp(-neg_log_probs_sum)
    probs_malignant=1 - probs_benign
    output=T.concatenate([probs_benign,probs_malignant],axis=1)
       
    return output
