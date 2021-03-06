import os
import numpy as np
from skimage import io
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Lambda
from keras.optimizers import SGD,Adam
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import utils
from callbacks.lossaccf2 import LossAccF2History
from helpers.dfgen import DFGen


#
# DEFAULTS
#
PROJECT_NAME='planet'
DATA_ROOT=os.environ.get('DATA')
WEIGHT_ROOT=os.environ.get('WEIGHTS')
DATA_DIR=f'{DATA_ROOT}/{PROJECT_NAME}'
WEIGHT_DIR=f'{WEIGHT_ROOT}/{PROJECT_NAME}'
OUTPUT_DIR='out'
HISTORY_DIR=f'{OUTPUT_DIR}/history'
TIF_BATCH_INPUT_SHAPE=(None,256,256,4)
JPG_BATCH_INPUT_SHAPE=(None,256,256,3)
TARGET_DIM=17
DEFAULT_OPT='adam'
DEFAULT_DR=0.5
DEFAULT_LOSS_FUNC='binary_crossentropy'
DEFAULT_METRICS=['accuracy']
DEFAULT_HISTORY=LossAccF2History
LR_REDUCER=ReduceLROnPlateau()    




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

    def __init__(self,
            batch_input_shape=None,
            image_ext='tif',
            optimizer=DEFAULT_OPT,
            loss_func=DEFAULT_LOSS_FUNC,
            target_dim=TARGET_DIM,
            metrics=DEFAULT_METRICS,
            auto_compile=True):
        self.image_ext=image_ext
        self.batch_input_shape=batch_input_shape or self._default_batch_input_shape()
        self.optimizer=optimizer
        self.loss_func=loss_func
        self.target_dim=target_dim
        self.metrics=metrics
        self.auto_compile=auto_compile
        self.history=None
        self._model=None

    def load_weights(self,pdata):
        # We may want to allow this to pass the version number 
        self.model().load_weights(f'{self._weight_path(pdata)}/sz{pdata.train_size}_tags{pdata.tags_to_string()}_v{pdata.version}.hdf5')


    def save_weights(self,pdata):
        # We may want to allow this to pass the version number
        self.model().save_weights(f'{self._weight_path(pdata)}/sz{pdata.train_size}_tags{pdata.tags_to_string()}_v{pdata.version}.hdf5')


    def model(self):
        print("ERROR[MODEL_BASE]: MUST OVERWRITE .model()")
        if not self._model:
            self._model=Sequential()
            if self.auto_compile: self._model.compile()
        return self._model


    def compile(self,loss_func=None,optimizer=None,metrics=None,reset_defaults=True):
        if loss_func:
            if reset_defaults: self.loss_func=loss_func
        else:
            loss_func=self.loss_func
        if optimizer:
            if reset_defaults: self.optimizer=optimizer
        else:
            optimizer=self.optimizer
        if metrics:
            if reset_defaults: self.metrics=metrics
        else:
            metrics=self.metrics
        return self.model().compile(
            loss=loss_func, 
            optimizer=optimizer,
            metrics=metrics)


    def predict_image(self,
            name=None,
            file_ext=None,
            data_root=DATA_DIR,
            image_dir=None,
            image=None,
            return_image=False):
        if not image:
            image=io.imread(self._image_path(name,file_ext,data_root,image_dir))
        pred=self.model().predict(np.expand_dims(image, axis=0))
        if return_image:
            return pred, image
        else:
            return pred

    def predict_dir(self, 
            image_dir, 
            batch_size, 
            data_root=DATA_DIR):
        """ predicts all files in data_root/image_dir
        """

        abs_dir=f'{data_root}/{image_dir}'
        image_names=os.listdir(abs_dir)
        batch_gen=self._dir_batches(image_dir, batch_size, data_root=DATA_DIR)
        predictions=[]
        i=1
        while (i-1)*batch_size < len(image_names):
            image_name_batch=next(batch_gen)
            pred_batch=[self.model().predict(np.expand_dims(io.imread(image_name),axis=0)) for image_name in image_name_batch]
            predictions=predictions+pred_batch
            i+=1
        return dict(zip(image_names,predictions))




    def fit_gen(self,
            epochs=None,
            pdata=None,
            train_sz=None,
            valid_sz=None,
            train_gen=None,
            valid_gen=None,
            steps_per_epoch=10,
            batch_size=32,
            ndvi_images=False,
            history=DEFAULT_HISTORY,
            history_name=None,
            checkpoint_name=None,
            save_all_checkpoints=False,
            reduce_lr=True,
            callbacks=[]):
        """ call fit_generator 
            Args:
                -   if pdata (instance of <data.planent:PlanetData>) 
                    use params from pdata
                -   otherwise used passed params
                -   history_name:
                        saves two files:
                            - {history_name}.train.p
                            - {history_name}.valid.p
                -   checkput_name:
                        saves weights after each epoch. file path is
                        {WEIGHT_DIR}/{checkpoint_name}.{epoch}-{loss}.hdf5
                -   number of images trained is epochs * batch_size * steps_per_epoch
        """
        if pdata:
            if not train_sz: train_sz=pdata.train_size
            if not valid_sz: valid_sz=pdata.valid_size
            train_gen=DFGen(
                dataframe=pdata.train_df,image_ext=self.image_ext,batch_size=batch_size,ndvi_images=ndvi_images)
            valid_gen=DFGen(
                dataframe=pdata.valid_df,image_ext=self.image_ext,batch_size=batch_size,ndvi_images=ndvi_images)

        if history:
            path=f'{HISTORY_DIR}/{history_name}'
            os.makedirs(os.path.dirname(path),exist_ok=True)
            self.history=history(save_path=path)
            callbacks.append(self.history)

        if checkpoint_name:
            if save_all_checkpoints:
                path=f'{WEIGHT_DIR}/{checkpoint_name}.{{epoch:02d}}-{{val_loss:.2f}}.hdf5'
            else:
                path=f'{WEIGHT_DIR}/{checkpoint_name}.hdf5'
            os.makedirs(os.path.dirname(path),exist_ok=True)
            callbacks.append(ModelCheckpoint(path,save_weights_only=True))

        if reduce_lr:
            callbacks.append(LR_REDUCER)
            
        validation_steps=valid_sz/batch_size

        return self.model().fit_generator(
            generator=train_gen,
            validation_data=valid_gen,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=self.VERBOSE)


    def _default_batch_input_shape(self):
        if self.image_ext=='jpg': 
            return JPG_BATCH_INPUT_SHAPE
        else: 
            return TIF_BATCH_INPUT_SHAPE

    def _image_path(self,name=None,file_ext=None,data_root=DATA_DIR,image_dir=None):
        fpath=f'{data_root}'
        if image_dir: fpath=f'{fpath}/{image_dir}'
        fpath=f'{fpath}/{name}'
        if file_ext: fpath=f'{fpath}.{file_ext}'
        return fpath


    def _dir_batches(self, image_dir, batch_size, data_root=DATA_DIR):
        """ Inputs: directpry and batch_size
            Returns: generator yieling batches of image-pathnames
        """
        abs_dir=f'{data_root}/{image_dir}'
        image_names=os.listdir(abs_dir)
        image_names=[self._image_path(image_name, data_root=DATA_DIR, image_dir=image_dir) for image_name in image_names]
        l = len(image_names)
        for i in range(0, l, batch_size):
            yield image_names[i:min(i + batch_size, l)]


    def _weight_path(self,pdata):
        tag_weight_path=f'{WEIGHT_DIR}/tags_{pdata.tags_to_string()}'
        if not os.path.isdir(tag_weight_path):
            os.mkdir(tag_weight_path)
        return tag_weight_path







