import math

def gen_params(train_size,valid_size,nb_epochs=5):
    """ Generator Params
        returns data for fit generator
        Args:
            train_size: <int> size of training dataset
            valid_size: <int> size of validation dataset
            nb_epochs: <int> number of epochs
        return:
            nb_epochs,steps_per_epoch, validation_steps
    """
    s=math.floor(train_size/nb_epochs)
    vs=math.floor(valid_size/nb_epochs)
    return nb_epochs,s,vs
