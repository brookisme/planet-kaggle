import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
import numpy as np

from models.base import MODEL_BASE

from activations.stepfunc import Stepfunc

#########################################################################################
#
# BASED OFF: https://www.kaggle.com/ekami66/step-by-step-solution-with-keras-0-89-on-lb
#
#########################################################################################
"""

    # 
    # BLOCKS
    #
    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))
        self.classifier.add(Conv2D(16, (2, 2), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))

    def add_flatten_layer(self):
        self.classifier.add(Flatten())


    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(256, activation='relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    # 
    # NETWORK
    #
    classifier = AmazonKerasClassifier()
    classifier.add_conv_layer(img_resize)
    classifier.add_flatten_layer()
    classifier.add_ann_layer(len(y_map))


    --------- NOTES ---------
    - i left in the zero padding
    - i'm going to start with the cos_distance loss func

"""

#
# COMMON BLOCKS
#
def ConvBlock(model,filters,layers=1):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model



def FCBlock(model,output_dim=256,dr=0.5):
    model.add(Dense(output_dim, activation='relu'))
    model.add(Dropout(dr))
    return model



class EKAMI(MODEL_BASE):
    def model(self):
        if not self._model:
            self._model=Sequential()
            self._model.add(BatchNormalization(batch_input_shape=self.batch_input_shape))
            self._model=ConvBlock(self._model,32)
            self._model=ConvBlock(self._model,64)
            self._model=ConvBlock(self._model,16)
            self._model.add(Flatten())
            self._model=FCBlock(self._model)
            self._model=FCBlock(self._model,512)
            self._model.add(Dense(self.target_dim, activation='sigmoid'))
            self._model.compile(loss=self.loss_func, 
                  optimizer=self.optimizer,
                  metrics=self.metrics)
        return self._model



class EKPLUS(MODEL_BASE):
    def model(self):
        if not self._model:
            self._model=Sequential()
            self._model.add(BatchNormalization(batch_input_shape=self.batch_input_shape))
            self._model=ConvBlock(self._model,32)
            self._model=ConvBlock(self._model,64)
            self._model=ConvBlock(self._model,16)
            self._model.add(Flatten())
            self._model=FCBlock(self._model)
            self._model=FCBlock(self._model,512)
            self._model.add(Dense(self.target_dim, activation='sigmoid'))
            self._model.add(Dense(self.target_dim))
            self._model.add(Stepfunc())
            self._model.compile(loss=self.loss_func, 
                  optimizer=self.optimizer,
                  metrics=self.metrics)
        return self._model



