import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers.core import Lambda
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dense
from keras.layers.merge import Concatenate
from keras.models import Model
import numpy as np
from models.base import MODEL_BASE



#
# DEFAULTS
#
DEFAULT_CONV_ACTIVATION='relu'
DEFAULT_FC_ACTIVATION='relu'
DEFAULT_OUTPUT_ACTIVATION='sigmoid'
DEFAULT_CONV_LAYERS=[
    (32,[3]),
    (64,[3]),
    (128,[3])]
DEFAULT_FC_LAYERS=[
    256,
    512]



#
# FLEX MODEL
#
class AF_BASE(MODEL_BASE):
    #
    # INTERNAL: BLOCKS
    #
    def _conv_block(self,x,filters,layers=[3],pool_size=(2,2)):
        for size in layers:
            x=ZeroPadding2D((size-2, size-2))(x)
            x=Conv2D(filters, (size,size), activation=self.conv_activation)(x)
        if self.batch_norm: x=BatchNormalization()(x)
        x=MaxPooling2D(pool_size=pool_size)(x)
        return x


    def _fc_block(self,x,output_dim=256,dr=0.5):
        x=Dense(output_dim, activation=self.fc_activation)(x)
        if self.batch_norm: x=BatchNormalization()(x)
        x=Dropout(dr)(x)
        return x



class Flex(AF_BASE):

    def __init__(self,
            lmbda=None,
            batch_norm=False,
            conv_layers=DEFAULT_CONV_LAYERS,
            fc_layers=DEFAULT_FC_LAYERS,
            conv_activation=DEFAULT_CONV_ACTIVATION,
            fc_activation=DEFAULT_FC_ACTIVATION,
            output_activation=DEFAULT_OUTPUT_ACTIVATION,
            **kwargs):
        self.lmbda=lmbda
        self.batch_norm=batch_norm
        self.conv_layers=conv_layers
        self.fc_layers=fc_layers
        self.conv_activation=conv_activation
        self.fc_activation=fc_activation
        self.output_activation=output_activation
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
                x=self._conv_block(x,filters,sizes)
            x=Flatten()(x)
            for output_dim in self.fc_layers:
                x=self._fc_block(x,output_dim=output_dim)
            outputs=Dense(self.target_dim, activation=self.output_activation)(x)
            self._model=Model(inputs=inputs,outputs=outputs)
            if self.auto_compile: self.compile()
        return self._model




class Combo(AF_BASE):
    """ Model from Models:
        - takes list of input models and sets all layers to have trainable=False
        - concatenates models
        - (optional) adds FC blocks
        - adds final output layer
        Usage:
            Assume you have 2 trained keras models m1, m2.
                For instance:
                    # model 1
                    flx1=Flex(...)
                    flx1.fit_gen(..)
                    m1=flx1.model()
                    # model 2
                    flx2=...
            # combo-model
            swell=Combo(input_models=[m1,m2])
            swell.fit_gen(..)
        Args:
            input_models: list of trained models
            batch_norm: True/False - perform batch_norm after blocks
            fc_layers: list of fc_layers between input_models concat and final layer
    """
    def __init__(self,
            input_models,
            batch_norm=False,
            fc_layers=DEFAULT_FC_LAYERS,
            fc_activation=DEFAULT_FC_ACTIVATION,
            output_activation=DEFAULT_OUTPUT_ACTIVATION,
            **kwargs):
        self.input_models=self._process_models(input_models)
        self.lmbda=lmbda
        self.batch_norm=batch_norm
        self.fc_layers=fc_layers
        self.fc_activation=fc_activation
        self.output_activation=output_activation
        super().__init__(**kwargs)


    def model(self):
        if not self._model:
            inputs=Input(batch_shape=self.batch_input_shape)
            premodels=[]
            for premodel in self.input_models:
                premodels.append(premodel(inputs))
            x=Concatenate()(premodels)
            for output_dim in self.fc_layers:
                x=self._fc_block(x,output_dim=output_dim)
            outputs=Dense(self.target_dim, activation=self.output_activation)(x)
            x=Model(inputs=inputs,outputs=outputs)

            self._model=Model(inputs=inputs,outputs=outputs)
            if self.auto_compile: self.compile()
        return self._model


    def _process_models(self,models):
        self.input_size=0
        for model in models:
            for layer in model.layers:
                layer.trainable=False
            self.input_size=self.input_size+model.output_shape[-1]
        return models

