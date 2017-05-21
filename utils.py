import math

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
