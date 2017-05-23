import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
import utils
import numpy as np
from skimage import io

from models.base import MODEL_BASE


TARGET_DIM=17
DEFAULT_OPT='adam'
DEFAULT_DR=0.5

#
# COMMON BLOCKS
#
def ConvBlock(model,layers,filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    return model



def FCBlock(model,dr=DEFAULT_DR,output_dim=256):
    model.add(Dense(output_dim, activation='relu'))
    model.add(Dropout(dr))
    return model



####################################################################
#
# DUMMYVGG: A DUMB MODEL TO GET US STARTED
#
####################################################################
class DummyVGG(MODEL_BASE):
    def model(self):
        if not self._model:
            self._model=Sequential()
            self._model.add(BatchNormalization(batch_input_shape=self.batch_input_shape))
            self._model=ConvBlock(self._model,2,32)
            self._model.add(Flatten())
            self._model=FCBlock(self._model)
            self._model.add(Dense(TARGET_DIM, activation='sigmoid'))
            self._model.compile(loss=self.loss_func, 
                  optimizer=self.optimizer,
                  metrics=['accuracy'])
        return self._model




####################################################################
#
# VGGARCH: VGG ARCHITECTURE
#
####################################################################
class VGGARCH(MODEL_BASE):
    LL_ACTIVATION='sigmoid'
    def model(self):
        if not self._model:
            self._model=Sequential()
            self._model.add(BatchNormalization(batch_input_shape=self.batch_input_shape))
            self._model=ConvBlock(self._model,2,32)
            self._model=ConvBlock(self._model,2,64)
            self._model=ConvBlock(self._model,2,128)
            self._model=ConvBlock(self._model,3,256)
            self._model=ConvBlock(self._model,3,512)
            self._model=ConvBlock(self._model,3,512)
            self._model.add(Flatten())
            self._model=FCBlock(self._model)
            self._model=FCBlock(self._model)
            self._model.add(Dense(TARGET_DIM, activation=self.LL_ACTIVATION))
            self._model.compile(loss=self.loss_func, 
                  optimizer=self.optimizer,
                  metrics=['accuracy'])
        return self._model


