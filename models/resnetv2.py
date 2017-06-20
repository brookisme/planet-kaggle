import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Activation, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Lambda
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dense
from keras.layers.merge import Concatenate
from keras.layers.merge import Add
from keras.models import Model
import numpy as np
from models.base import MODEL_BASE



class ResNet50_BASE(MODEL_BASE):
    #
    # INTERNAL: BLOCKS
    #

    DEFAULT_ID_KERNELS=[1,3,1]
    DEFAULT_ID_STRIDES=[1,1,1]
    DEFAULT_ID_PADDING=['valid','same','valid']

    def _identity_block(self,
        input_tensor, 
        filters, 
        kernel_sizes=DEFAULT_ID_KERNELS, 
        strides=DEFAULT_ID_STRIDES,
        padding=DEFAULT_ID_PADDING):
        """ The identity block has a shortcut which is just the identity
        """

        filters1, filters2, filters3 = filters
        kernel_size1, kernel_size2, kernel_size3= kernel_sizes
        strides1, strides2, strides3 = strides
        padding1, padding2, padding3 = padding

        x = Conv2D(filters=filters1, kernel_size=kernel_size1, strides=strides1, padding=padding1)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters2, kernel_size=kernel_size2, strides=strides2, padding=padding2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters3, kernel_size=kernel_size3, strides=strides3, padding=padding3)(x)
        x = BatchNormalization()(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        
        return x

    DEFAULT_CONV_KERNELS=[1,3,1,1]
    DEFAULT_CONV_STRIDES=[2,1,1,2]
    DEFAULT_CONV_PADDING=['valid','same','valid','valid']

    def _conv_block(self,
        input_tensor, 
        filters, 
        kernel_sizes=DEFAULT_CONV_KERNELS, 
        strides=DEFAULT_CONV_STRIDES,
        padding=DEFAULT_CONV_PADDING):
        """ The conv_block block has a shortcut which contains a Conv layer and batchnorm.
            Parameters must be chosen such that x and shortcut have identical shapes
            before the Add layer. This is achieved with strides4=strides1
        """

        kernel_size1, kernel_size2, kernel_size3, kernel_size4 = kernel_sizes
        filters1, filters2, filters3, filters4 = filters
        strides1, strides2, strides3, strides4 = strides
        padding1, padding2, padding3, padding4 = padding


        x = Conv2D(filters=filters1, kernel_size=kernel_size1, strides=strides1, padding=padding1)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters2, kernel_size=kernel_size2, strides=strides2, padding=padding2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters3, kernel_size=kernel_size3, strides=strides3, padding=padding3)(x)
        x = BatchNormalization()(x)

        shortcut = Conv2D(filters=filters4, kernel_size=kernel_size4, strides=strides4, padding=padding4)(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        
        return x



DEFAULT_OUTPUT_ACTIVATION='softmax'


class ResNet50(ResNet50_BASE):


    def __init__(self,
            output_activation=DEFAULT_OUTPUT_ACTIVATION,
            **kwargs):

        self.output_activation=output_activation
        super().__init__(**kwargs)

    def model(self):
        if not self._model:
            inputs=Input(batch_shape=self.batch_input_shape)
            x=BatchNormalization()(inputs)

            # x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
            # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

            x = self._conv_block(x, filters=[64, 64, 256, 256], strides=[1,1,1,1])
            x = self._identity_block(x, filters=[64, 64, 256])
            x = self._identity_block(x, filters=[64, 64, 256])

            x = self._conv_block(x,filters=[128, 128, 512, 512])
            x = self._identity_block(x, filters=[128, 128, 512])
            x = self._identity_block(x, filters=[128, 128, 512])
            x = self._identity_block(x, filters=[128, 128, 512])

            x = self._conv_block(x, filters=[256, 256, 1024, 1024])
            x = self._identity_block(x, filters=[256, 256, 1024])
            x = self._identity_block(x, filters=[256, 256, 1024])
            x = self._identity_block(x, filters=[256, 256, 1024])
            x = self._identity_block(x, filters=[256, 256, 1024])
            x = self._identity_block(x, filters=[256, 256, 1024])

            x = self._conv_block(x, filters=[512, 512, 2048, 2048])
            x = self._identity_block(x, filters=[512, 512, 2048])
            x = self._identity_block(x, filters=[512, 512, 2048])

            x = AveragePooling2D(pool_size=(7, 7))(x)

            x = Flatten()(x)
            outputs = Dense(self.target_dim, activation=self.output_activation)(x)

            self._model=Model(inputs=inputs,outputs=outputs)
            if self.auto_compile: self.compile()

        return self._model




