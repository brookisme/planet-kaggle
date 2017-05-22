import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Lambda
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
import utils

#
# DEFAULTS
#
BANDS=4
BATCH_INPUT_SHAPE=(None,256,256,BANDS)
TARGET_DIM=17
DEFAULT_OPT=Adam
DEFAULT_LR=0.01
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



def FCBlock(model,dr=DEFAULT_DR,output_dim=200):
    model.add(Dense(output_dim, activation='relu'))
    model.add(Dropout(dr))
    return model




####################################################################
#
# MODEL_BASE: BASE CLASS FOR MODELS
#   Args:
#       batch_input_shape (None,w,h,ch)
#       optimizer (SGD, Adam, ...)
# must provide a self.model() method that uses the args above
#
####################################################################
class MODEL_BASE(object):
    VERBOSE=1

    def __init__(self,batch_input_shape=BATCH_INPUT_SHAPE,optimizer='adam'):
        self.batch_input_shape=batch_input_shape
        self.optimizer=optimizer
        self._model=None


    def fit_gen(self,train_sz,valid_sz,epochs,
            train_gen=None,
            valid_gen=None,
            sample_pct=1.0):
        nb_epochs,steps,validation_steps=utils.gen_params(
            train_sz,valid_sz,epochs,sample_pct)
        self.model().fit_generator(
            generator=train_gen, 
            validation_data=valid_gen,
            steps_per_epoch=steps,
            validation_steps=validation_steps,
            epochs=epochs, 
            verbose=self.VERBOSE)



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
            self._model.compile(loss='binary_crossentropy', 
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
            self._model.compile(loss='binary_crossentropy', 
                  optimizer=self.optimizer,
                  metrics=['accuracy'])
        return self._model


