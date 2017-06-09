import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers.core import Lambda
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dense, Merge
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
    (16,[3])]
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
            x=ZeroPadding2D((1, 1))(x)
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
    # --> 122                 premodels.append(Model(inputs=inputs, outputs=model))
    # TypeError: Output tensors to a Model must be Keras tensors. 
    #             Found: <keras.engine.training.Model object at 0x7fe20d013f28>
    def __init__(self,
            input_models,
            lmbda=None,
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
            x=Merge(self.input_models)(inputs)
            for output_dim in self.fc_layers:
                x=self._fc_block(x,output_dim=output_dim)
            outputs=Dense(self.target_dim, activation=self.output_activation)(x)
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

