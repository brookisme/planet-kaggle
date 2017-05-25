import os
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
import utils
import numpy as np
from skimage import io

from models.base import MODEL_BASE

#########################################################################################
#
# BASED OFF: https://www.kaggle.com/northomas/first-try-using-keras-cnn
#
#########################################################################################
"""

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())



model = Sequential()

model.add(Conv2D(48, (8, 8), strides=(2, 2), input_shape=INPUT_SHAPE, activation='elu'))
model.add(BatchNormalization())

model.add(Conv2D(64, (8, 8), strides=(2, 2), activation='elu'))
model.add(BatchNormalization())

model.add(Conv2D(96, (5, 5), strides=(2, 2), activation='elu'))
model.add(BatchNormalization())

model.add(Conv2D(96, (3, 3), activation='elu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.3))

model.add(Dense(256, activation='elu'))
model.add(BatchNormalization())

model.add(Dense(64, activation='elu'))
model.add(BatchNormalization())

model.add(Dense(n_classes, activation='sigmoid'))

    
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[fbeta, 'accuracy']
)


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



class NORTHOM(MODEL_BASE):
    def model(self):
        if not self._model:
            self._model=Sequential()
            # ...
            self._model.compile(loss=self.loss_func, 
                  optimizer=self.optimizer,
                  metrics=self.metrics)
        return self._model



