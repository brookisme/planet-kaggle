import math
import numpy as np
from keras import backend as K
from sklearn.metrics import fbeta_score


def gen_params(train_size,valid_size,epochs=5,sample_pct=1.0):
    """ Generator Params
        returns data for fit generator
        Args:
            train_size: <int> size of training dataset
            valid_size: <int> size of validation dataset
            epochs: <int> number of epochs
        return:
            epochs,steps_per_epoch, validation_steps
    """
    s=math.floor(sample_pct*train_size/epochs)
    vs=math.floor(sample_pct*valid_size/epochs)
    return epochs,s,vs


def cos_distance(a,b,return_type=None):
    """ Cosine Distance
        returns cosine-distance between two vectors
        Args:
            a: <list|np.array|K.tensor> first vector
            b: <list|np.array|K.tensor> second vector
            return_type: <str> if 'float' it will call eval on tensor
        return:
            epochs,steps_per_epoch, validation_steps
    """  
    if not K.is_keras_tensor(a): a=K.variable(a)
    if not K.is_keras_tensor(b): b=K.variable(b)
    vec_a=K.l2_normalize(a,axis=-1)
    vec_b=K.l2_normalize(b,axis=-1)
    c_dist=K.mean(1-K.sum(vec_a*vec_b,axis=-1))
    if return_type=='float': c_dist=K.eval(c_dist)
    return c_dist 


def f2_score(y_true, y_pred,average='samples'):
    """ F2 Score
        description: 
            https://www.kaggle.com/c/planet-understanding-the-amazon-from-space#evaluation
        method: 
            https://www.kaggle.com/anokas/fixed-f2-score-in-python
    """
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average=average)


def k_f2(y_true, y_pred, threshold_shift=0,return_type=None):
    #
    #
    # SEE BELOW FOR CORRECT/NOTCORRECTNESS
    #
    #
    beta = 2
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_squared = beta ** 2
    f2val=(beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    if return_type=='float': f2val=K.eval(f2val)
    return f2val


def k_f2_micro(y_true, y_pred,return_type=None):
    #
    # micro is the global average 
    # which happes to be correct for 1 pred
    # this is incorrect for batches!
    #
    y_pred = K.clip(y_pred, 0, 1)
    tp = K.sum(y_true * y_pred) + K.epsilon()
    fp = K.sum((1-y_true)*y_pred)
    fn = K.sum(y_true*(1-y_pred))
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f2val= 5*precision*recall/((4* precision) + recall + K.epsilon())
    if return_type=='float': f2val=K.eval(f2val)
    return f2val


def k_f2_loss(y_true, y_pred):
    return 1-k_f2_micro(y_true, y_pred)


def step_act(value):
    return K.round(value)



