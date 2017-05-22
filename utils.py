import math
from keras import backend as K



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
    if return_type=='float':
        return K.eval(c_dist)
    else:
        return c_dist 