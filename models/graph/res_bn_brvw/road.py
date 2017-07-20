import os
import main
from dfgen import DFGen
from keras.models import Model
from keras.layers import Dense, Dropout
import argparse

#
# SETUP
#


tags=['road']


ACTIVATION='sigmoid'
COMPLEXITY=[1024,512]
WEIGHTS=None
RUN_START=0
REQ_PCT=None
AUGMENT=True


MODEL_NAME='_'.join(tags)
train_csv=f'train_{MODEL_NAME}.csv'
valid_csv=f'valid_{MODEL_NAME}.csv'
TRAIN_BATCH_SIZE=64 #main.TRAIN_BATCH_SIZE
VALID_BATCH_SIZE=main.VALID_BATCH_SIZE
TRAIN_STEPS=200
VALID_STEPS=100


#
# GENERATORS
#
def create_csv(gen,path,tags):
    gen.reduce_columns(*tags,others=False)
    if REQ_PCT:
        gen.require_values(REQ_PCT)
    gen.save(path)


if not os.path.isfile(train_csv):
    create_csv(main.train_gen,train_csv,tags)


if not os.path.isfile(valid_csv):
    create_csv(main.valid_gen,valid_csv,tags)


train_gen=DFGen(
    csv_file=train_csv,
    csv_sep=',',
    batch_size=TRAIN_BATCH_SIZE,
    lambda_func=main.norm_brvw,
    tags=tags,
    augment=AUGMENT)


valid_gen=DFGen(
    csv_file=valid_csv,
    csv_sep=',',
    batch_size=VALID_BATCH_SIZE,
    lambda_func=main.norm_brvw,
    tags=tags,
    augment=AUGMENT)


train_gen.dataframe.head()
valid_gen.dataframe.head()

train_gen.size
valid_gen.size



"""
#
# MODEL-SETUP
#
"""
def set_trainable(kgmodel,trainable=False):
    for layer in kgmodel.model().layers: 
        layer.trainable=trainable


kg_model=main.get_kg_model()
kg_model.load_weights(main.BASE_WEIGHTS)
set_trainable(kg_model)
inputs=kg_model.model().input


if COMPLEXITY:
    kg_model.model().layers.pop()
    x=kg_model.model().layers[-1].output
    for complexity in COMPLEXITY:
        x=Dense(complexity, activation='relu')(x)
        x=Dropout(0.5)(x)
else:
    inputs=kg_model.model().input
    x=kg_model.model().layers[-1].output


outputs=Dense(len(tags), activation=ACTIVATION)(x)
kg_model._model=Model(inputs=inputs,outputs=outputs)


if WEIGHTS:
    kg_model.load_weights(WEIGHTS)


kg_model.compile()
kg_model.model().summary() 


#
# MODEL-RUN
#
def save_model(name):
    kg_model.model().save(f'models/{name}.hdf5')


def run_model(n,ident='A'):
    for i in range(RUN_START,RUN_START+n):
        run_name=f'{MODEL_NAME}_{ident}_{i}'
        if (RUN_START-i): epochs=5
        else: epochs=5
        print(f'\n\n{run_name}:')
        kg_model.fit_gen(
             epochs=epochs,
             train_gen=train_gen,
             train_steps=TRAIN_STEPS,
             validation_gen=valid_gen,
             validation_steps=VALID_STEPS,
             history_name=run_name,
             checkpoint_name=run_name)
        save_model(run_name)


run_model(3,"x".join(map(str,COMPLEXITY)) or 'linearmap')



