from __future__ import print_function
import keras
from keras.models import Sequential
from keras import layers
from keras.datasets import mnist
from keras import backend as K


class ReReLU(layers.Layer):
    ''' A Relu-like activation function for binary output
                 ___
        ie:  ___/
    '''
    def __init__(self,slope=1,x_intercept=0,**kwargs):
        self.slope=slope
        self.x_intercept=x_intercept
        super().__init__(**kwargs)


    def call(self,inputs):
        inputs=K.expand_dims(inputs, axis=0)
        zeros=K.zeros_like(inputs)
        ones=K.ones_like(inputs)
        shape=K.shape(inputs)
        relu_inputs=self.slope*(inputs-(self.x_intercept*ones))
        relu=K.max(self._bound_cat(zeros,relu_inputs),axis=0)
        rerelu=K.min(self._bound_cat(ones,relu,shape),axis=0)
        return rerelu


    #
    # INTERNAL
    #
    def _bound_cat(self,bounds,inputs,shape=None):
        if shape is not None: inputs=K.reshape(inputs,shape)
        return K.concatenate([bounds,inputs],axis=0)