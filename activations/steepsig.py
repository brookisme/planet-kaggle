from __future__ import print_function
import keras
from keras.models import Sequential
from keras import layers
from keras.datasets import mnist
from keras import backend as K


class Stepfunc(layers.Layer):
    '''A simple step function
    -   for use after last dense layer to threshold values to 0 or 1
    -   We use round (aka threshold of 0.5) but this shouldn't matter
        as the weights will decide what the best output value is.
    '''
    def __init__(self,n=6,t=0.5,**kwargs):
        self.N=n
        self.T=t
        super(Stepfunc, self).__init__(**kwargs)


    def call(self, inputs):
        print("Stepfunc--",self.N,self.T)
        mid=self.T * K.ones_like(inputs)
        outputs=1/(1+K.exp(-self.N*(inputs-mid)))
        return outputs

