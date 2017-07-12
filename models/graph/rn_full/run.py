import os
from pprint import pprint
from dfgen import DFGen
from kgraph.functional import RESNET as R
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau

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
sgd=SGD(lr=0.1,momentum=0.9,decay=0.0001)
callbacks=[ReduceLROnPlateau(patience=5)]
train_batch_size=32
valid_batch_size=16

#
# GENERATORS
#
PKR=os.environ['PKR']
train_csv_path=f'{PKR}/datacsvs/train.csv'
valid_csv_path=f'{PKR}/datacsvs/valid.csv'
train_gen=DFGen(csv_file=train_csv_path,csv_sep=',',batch_size=train_batch_size)
valid_gen=DFGen(csv_file=valid_csv_path,csv_sep=',',batch_size=valid_batch_size)
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
        'optimizer': sgd
    },
    'inputs': {
        'batch_shape':(None,256,256,4),
    },
    'output': { 
        'units': 17,
        'activation': 'sigmoid' 
    }
}
kgres=R(graph)
# inspect
pprint(kgres.graph)
kgres.model().summary()


#
# MODEL-RUN
#
RUN_NAME='rn17-0712-2'


kgres.fit_gen(
   epochs=60,
   train_gen=train_gen,
   train_steps=200,
   validation_gen=valid_gen,
   validation_steps=100,
   history_name=RUN_NAME,
   checkpoint_name=RUN_NAME,
   callbacks=callbacks)

