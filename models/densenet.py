import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Activation, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Lambda
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dense
from keras.layers.merge import Add, Concatenate
from keras.regularizers import l2
import keras.backend as K

from keras.models import Model
import numpy as np
from models.base import MODEL_BASE



class DenseNet_BASE(MODEL_BASE):

    DEFAULT_CONV_FILTER=64
    DEFAULT_CONV_KERNEL=3
    DEFAULT_CONV_STRIDE=1
    DEFAULT_CONV_PADDING='same'
    DEFAULT_CONV_DROPOUT=.3
    DEFAULT_DENSE_LAYERS=3




    def _conv_block(self,
        input_tensor, 
        filters=DEFAULT_CONV_FILTER, 
        kernel_size=DEFAULT_CONV_KERNEL, 
        strides=DEFAULT_CONV_STRIDE,
        padding=DEFAULT_CONV_PADDING,
        dropout=DEFAULT_CONV_DROPOUT):
        """ The conv_block block has a shortcut which contains a Conv layer and batchnorm.
            Parameters must be chosen such that x and shortcut have identical shapes
            before the Add layer. This is achieved with strides4=strides1
        """

        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x=Dropout(dropout)(x)

        return x


    def _denseblock(self,x, nb_layers=DEFAULT_DENSE_LAYERS):

        list_feat = [x]
        for i in range(nb_layers):
            x = self._conv_block(x)
            list_feat.append(x)
            x = Concatenate()(list_feat)

        return x



class DenseNet(DenseNet_BASE):

    DEFAULT_OUTPUT_ACTIVATION='softmax'

    def __init__(self,
            output_activation=DEFAULT_OUTPUT_ACTIVATION,
            **kwargs):

        self.output_activation=output_activation
        super().__init__(**kwargs)

    
    def model(self):
        if not self._model:
            inputs=Input(batch_shape=self.batch_input_shape)
            x=BatchNormalization()(inputs)

            x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1,1),padding='same')(x)

            x = self._denseblock(x)

            x = self._conv_block(x)
            x = AveragePooling2D(pool_size=(2,2), strides=(2, 2))(x)

            x = self._denseblock(x)

            x = self._conv_block(x)
            x = AveragePooling2D(pool_size=(2,2), strides=(2, 2))(x)

            x = self._denseblock(x)

            x=BatchNormalization()(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=(2,2), strides=(2, 2))(x)

            x = Flatten()(x)
            outputs = Dense(self.target_dim, activation=self.output_activation)(x)

            self._model=Model(inputs=inputs,outputs=outputs)

            if self.auto_compile: 
                self.compile()

        return self._model






