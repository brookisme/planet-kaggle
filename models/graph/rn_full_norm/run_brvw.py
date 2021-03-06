import os
from pprint import pprint
from dfgen import DFGen
from kgraph.functional import RESNET as R
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
import numpy as np
""" NOTES
        OPTIMIZER:
                Keras SGD DEFAULTS: SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
                RESNET SGD:
                        lr=0.1
                        weight decay=0.0001
                        momentum=0.9


        LR-REDUCER:
                Keras DEFAULTS: ReduceLROnPlateau(
                        monitor='val_loss', factor=0.1, patience=10, 
                        verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
                RESNET:
                        lr is divided by 10 when the error plateaus
                        ~ keep defaults

        * the models are trained for up to 60 × 10^4
        * mini-batch size of 256
"""
#
# SETUP
#
# sgd=SGD(lr=0.1,momentum=0.9,decay=0.0001)
# callbacks=[ReduceLROnPlateau(patience=5)]
#
#
adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#
#
BATCH_SHAPE=(None,256,256,4)
train_batch_size=32
valid_batch_size=16


R_MEAN=3072.1367933921815
R_STDEV=450.97146444273375

G_MEAN=4268.506753740692
G_STDEV=409.536961252693

B_MEAN=4969.024985050201
B_STDEV=394.0761093545407

N_MEAN=6400.041219993591
N_STDEV=856.3157545106753

V_MEAN=1.0444824034038314
V_STDEV=0.6431990734340584

W_MEAN=1.1299844489632986
W_STDEV=0.9414969419163138


MEANS=np.array([R_MEAN,B_MEAN,G_MEAN,N_MEAN])
STDEVS=np.array([R_STDEV,B_STDEV,G_STDEV,N_STDEV])


BRVW_MEANS=np.array([R_MEAN,B_MEAN,V_MEAN,W_MEAN])
BRVW_STDEVS=np.array([R_STDEV,B_STDEV,V_STDEV,W_STDEV])


def norm(img):
    img=(img-MEANS)/STDEVS
    return img[:]


def bgrn_to_brvw_img(img):
    g,r,nir=img[:,:,1], img[:,:,2], img[:,:,3]
    ndwi=(nir-g)/(nir+g)
    del g
    ndvi=(nir-r)/(nir+r)
    del nir
    return np.dstack([img[:,:,0],r,ndvi,ndwi])


def norm_brvw(img):
    img=bgrn_to_brvw_img(img)
    img=(img-BRVW_MEANS)/BRVW_STDEVS
    return img


#
# GENERATORS
#
PKR=os.environ['PKR']
AUG_N=0

train_csv_path=f'{PKR}/datacsvs/aug_train.{AUG_N}.csv'
valid_csv_path=f'{PKR}/datacsvs/aug_valid.{AUG_N}.csv'
train_gen=DFGen(csv_file=train_csv_path,csv_sep=',',batch_size=train_batch_size,lambda_func=norm_brvw)
valid_gen=DFGen(csv_file=valid_csv_path,csv_sep=',',batch_size=valid_batch_size,lambda_func=norm_brvw)
# inspect
train_gen.dataframe.sample(3)


#
# MODEL
# 
graph={
      
  'meta': {
                'network_type': 'RESNET'
        },
        'compile': {
                'loss_func': 'binary_crossentropy',
                'metrics': ['accuracy'],
                'optimizer': 'adam'
        },
        'inputs': {
                'batch_shape':BATCH_SHAPE,
        },
        'input_layers': [
            {   
                'type':'Conv2D',
                'filters':64,
                'kernel_size':7,
                'strides':2,
                'padding':'same'
            },
            {
                'type': 'BatchNormalization'
            },
            {
                'type': 'Activation',
                'activation':'relu'
            },
            {
                'type':'MaxPooling2D',
                'pool_size':3,
                'strides':2,
                'padding':'same'
            }
        ],
        'output': { 
                'units': 17,
                'activation': 'sigmoid' 
        }
}
kgres=R(graph)
# inspect
pprint(kgres.graph)
kgres.load_weights('rn17-brvw-aug0-1.hdf5')
# kgres.model().summary()


#
# MODEL-RUN
#
RUN_NAME='rn17-brvw-aug0-2'


kgres.fit_gen(
     epochs=20,
     train_gen=train_gen,
     train_steps=200,
     validation_gen=valid_gen,
     validation_steps=100,
     history_name=RUN_NAME,
     checkpoint_name=RUN_NAME)



RUN_NAME='rn17-brvw-aug0-3'


kgres.fit_gen(
     epochs=20,
     train_gen=train_gen,
     train_steps=200,
     validation_gen=valid_gen,
     validation_steps=100,
     history_name=RUN_NAME,
     checkpoint_name=RUN_NAME)


RUN_NAME='rn17-brvw-aug0-4'


kgres.fit_gen(
     epochs=20,
     train_gen=train_gen,
     train_steps=200,
     validation_gen=valid_gen,
     validation_steps=100,
     history_name=RUN_NAME,
     checkpoint_name=RUN_NAME)


RUN_NAME='rn17-brvw-aug0-5'


kgres.fit_gen(
     epochs=20,
     train_gen=train_gen,
     train_steps=200,
     validation_gen=valid_gen,
     validation_steps=100,
     history_name=RUN_NAME,
     checkpoint_name=RUN_NAME)


RUN_NAME='rn17-brvw-aug0-6'


kgres.fit_gen(
     epochs=20,
     train_gen=train_gen,
     train_steps=200,
     validation_gen=valid_gen,
     validation_steps=100,
     history_name=RUN_NAME,
     checkpoint_name=RUN_NAME)




kgres.load_weights('rn17-brvw-aug0-1.hdf5')
adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
kgres.compile(optimizer=adam)


RUN_NAME='rn17-brvw-opt0001-1'


kgres.fit_gen(
     epochs=20,
     train_gen=train_gen,
     train_steps=200,
     validation_gen=valid_gen,
     validation_steps=100,
     history_name=RUN_NAME,
     checkpoint_name=RUN_NAME)



