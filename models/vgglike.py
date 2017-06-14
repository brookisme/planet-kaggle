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
from skimage import io
from PIL import Image

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



def FCBlock(model,dr=DEFAULT_DR,output_dim=4096):
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
                  metrics=self.metrics)
        return self._model




####################################################################
#
# VGGARCH: VGG ARCHITECTURE
#
####################################################################


PROJECT_NAME='planet'
WEIGHT_ROOT=os.environ.get('WEIGHTS')
WEIGHT_DIR=f'{WEIGHT_ROOT}/{PROJECT_NAME}'

class VGGARCH(MODEL_BASE):
    def model(self):
        if not self._model:
            self._model=Sequential()
#            self._model.add(BatchNormalization(batch_input_shape=self.batch_input_shape))
            self._model.add(Lambda(self.vgg_preprocess, input_shape=(256,256,3), output_shape=(224,224,3)))
            self._model=ConvBlock(self._model,2,64)
            self._model=ConvBlock(self._model,2,128)
            self._model=ConvBlock(self._model,3,256)
            self._model=ConvBlock(self._model,3,512)
            self._model=ConvBlock(self._model,3,512)
            self._model.add(Flatten())
            self._model=FCBlock(self._model)
            self._model=FCBlock(self._model)
            self._model.add(Dense(1000, activation='softmax'))
            self._model.compile(loss='categorical_crossentropy', 
                  optimizer=self.optimizer,
                  metrics=['accuracy'])
            self._model.load_weights(f'{WEIGHT_DIR}/VGG/vgg16.h5')
            self._model.layers.pop()
            for layer in self._model.layers: layer.trainable=False
            self._model.add(Dense(self.target_dim, activation='softmax'))
            self._model.compile(loss='categorical_crossentropy', 
                optimizer=self.optimizer,
                metrics=['accuracy'])

        return self._model

    def vgg_preprocess(self,x):

        vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
        x = ktf.image.resize_images(x-vgg_mean, (224, 224))
        return x



