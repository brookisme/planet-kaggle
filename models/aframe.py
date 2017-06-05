import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers.core import Lambda
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from models.base import MODEL_BASE
from activations.stepfunc import Stepfunc


#
# COMMON BLOCKS
#
def ConvBlock(x,filters,layers=1,batch_norm=False):
    for i in range(layers):
        x=ZeroPadding2D((1, 1))(x)
        x=Conv2D(filters, (3, 3), activation='relu')(x)
    if batch_norm: x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    return x



def FCBlock(x,output_dim=256,batch_norm=False,dr=0.5):
    x=Dense(output_dim, activation='relu')(x)
    if batch_norm: x=BatchNormalization()(x)
    x=Dropout(dr)(x)
    return x


class Flex(MODEL_BASE):
    def __init__(self,lmbda=None,batch_norm=False,**kwargs):
        self.lmbda=lmbda
        self.batch_norm=batch_norm
        super().__init__(**kwargs)


    def model(self):
        if not self._model:
            inputs=Input(batch_shape=self.batch_input_shape)
            if self.lmbda:
                x=Lambda(self.lmbda)(inputs)
                x=BatchNormalization()(x)
            else:
                x=BatchNormalization()(inputs)
            x=ConvBlock(x,32,batch_norm=self.batch_norm)
            x=ConvBlock(x,64,batch_norm=self.batch_norm)
            x=ConvBlock(x,16,batch_norm=self.batch_norm)
            x=Flatten()(x)
            x=FCBlock(x,batch_norm=self.batch_norm)
            x=FCBlock(x,512,batch_norm=self.batch_norm)
            outputs=Dense(self.target_dim, activation='sigmoid')(x)
            self._model=Model(inputs=inputs,outputs=outputs)
            if self.auto_compile: self.compile()
        return self._model

