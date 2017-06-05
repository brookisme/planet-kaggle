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

DEFAULT_CONV_LAYERS=[
    (32,[3]),
    (64,[3]),
    (16,[3])]


DEFAULT_FC_LAYERS=[
    256,
    512]


#
# COMMON BLOCKS
#



class Flex(MODEL_BASE):
    def __init__(self,
            lmbda=None,
            batch_norm=False,
            conv_layers=DEFAULT_CONV_LAYERS,
            fc_layers=DEFAULT_FC_LAYERS,
            **kwargs):
        self.lmbda=lmbda
        self.batch_norm=batch_norm
        self.conv_layers=conv_layers
        self.fc_layers=fc_layers
        super().__init__(**kwargs)


    def model(self):
        if not self._model:
            inputs=Input(batch_shape=self.batch_input_shape)
            if self.lmbda:
                x=Lambda(self.lmbda)(inputs)
                x=BatchNormalization()(x)
            else:
                x=BatchNormalization()(inputs)
            for filters,sizes in self.conv_layers:
                print("CONV:",filters,sizes)
                x=self._conv_block(x,filters,sizes)
            x=Flatten()(x)
            for output_dim in self.fc_layers:
                print("FC:",output_dim)
                x=self._fc_block(x,output_dim=output_dim)
            outputs=Dense(self.target_dim, activation='sigmoid')(x)
            self._model=Model(inputs=inputs,outputs=outputs)
            if self.auto_compile: self.compile()
        return self._model



    #
    # INTERNAL: BLOCKS
    #
    def _conv_block(self,x,filters,layers=[3],pool_size=(2,2)):
        for size in layers:
            x=ZeroPadding2D((1, 1))(x)
            x=Conv2D(filters, (size,size), activation='relu')(x)
        if self.batch_norm: x=BatchNormalization()(x)
        x=MaxPooling2D(pool_size=pool_size)(x)
        return x


    def _fc_block(self,x,output_dim=256,dr=0.5):
        x=Dense(output_dim, activation='relu')(x)
        if self.batch_norm: x=BatchNormalization()(x)
        x=Dropout(dr)(x)
        return x
