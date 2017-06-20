import os
import keras
from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Lambda
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf as ktf
import utils
import numpy as np

from models.base import MODEL_BASE


TARGET_DIM=17
DEFAULT_OPT='adam'
DEFAULT_DR=0.5
PROJECT_NAME='planet'
WEIGHT_ROOT=os.environ.get('WEIGHTS')
WEIGHT_DIR=f'{WEIGHT_ROOT}/{PROJECT_NAME}'

# VGG weights are downloaded from here
# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
VGG_WEIGHT_PATH=f'{WEIGHT_DIR}/VGG/vgg16.h5'
VGG_MEAN = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))

#
# COMMON BLOCKS
#
def ConvBlock(model,layers,filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    return model



def FCBlock(model,dr=DEFAULT_DR,output_dim=4096,activation='relu'):
    model.add(Dense(output_dim, activation=activation))
    model.add(Dropout(dr))
    return model



####################################################################
#
# VGG16: VGG16 ARCHITECTURE
#
####################################################################

class VGG16(MODEL_BASE):
    def model(self):
        if not self._model:
            self._model=Sequential()
            self._model.add(Lambda(self._vgg_preprocess, input_shape=(256,256,3), output_shape=(224,224,3)))
            self._model=ConvBlock(self._model,2,64)
            self._model=ConvBlock(self._model,2,128)
            self._model=ConvBlock(self._model,3,256)
            self._model=ConvBlock(self._model,3,512)
            self._model=ConvBlock(self._model,3,512)
            self._model.add(Flatten())

            self._model=FCBlock(self._model)
            self._model=FCBlock(self._model)
            self._model=FCBlock(self._model,dr=0,output_dim=1000,activation='softmax')

#            self._model.add(Dense(1000, activation='softmax'))
            self._model.load_weights(f'{VGG_WEIGHT_PATH}')
            self._model.compile(loss=self.loss_func, 
                optimizer=self.optimizer,
                metrics=self.metrics)

        return self._model

    def _vgg_preprocess(self,x):
        x = ktf.image.resize_images(x-VGG_MEAN, (224, 224))
        return x

####################################################################
#
# VGG16_FT: VGG16 ARCHITECTURE with a fine tuned final layer 
#           to give a softmax on a subset of TAGS
#
####################################################################

class VGG16_FT(MODEL_BASE):

    # def __init__(self,vggtrain=False):
    #     self.vggtrain=vggtrain

    def model(self):
        if not self._model:
            self._model=VGG16().model()
            self._model.layers.pop()
            # if not self.vggtrain:
            for layer in self._model.layers: layer.trainable=False
            self._model.add(Dense(self.target_dim, activation='softmax'))
            self._model.compile(loss=self.loss_func, 
                optimizer=self.optimizer,
                metrics=self.metrics)

        return self._model

