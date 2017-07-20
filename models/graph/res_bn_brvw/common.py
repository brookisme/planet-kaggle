import main
from dfgen import DFGen
from keras.models import Model
from keras.layers import Dense, Dropout
#
# SETUP
#

MODEL_NAME='common'
tags=['primary','agriculture','road','water','cultivation','habitation']
train_csv=f'train_{MODEL_NAME}.csv'
valid_csv=f'valid_{MODEL_NAME}.csv'


#
# GENERATORS
#
def create_csvs(tags):
    main.train_gen.reduce_columns(*tags,others=False)
    main.train_gen.save(train_csv)
    main.valid_gen.reduce_columns(*tags,others=False)
    main.valid_gen.save(valid_csv)

# create_csvs(tags)

train_gen=DFGen(
    csv_file=train_csv,
    csv_sep=',',
    batch_size=main.TRAIN_BATCH_SIZE,
    lambda_func=main.norm_brvw,
    tags=tags)
valid_gen=DFGen(
    csv_file=valid_csv,
    csv_sep=',',
    batch_size=main.VALID_BATCH_SIZE,
    lambda_func=main.norm_brvw,
    tags=tags)

train_gen.dataframe.head()
valid_gen.dataframe.head()



#
# MODEL-SETUP
#
kg_model=main.get_kg_model()
kg_model.load_weights(main.BASE_WEIGHTS)
kg_model.model().layers.pop()
for layer in kg_model.model().layers: 
    layer.trainable=False

inputs=kg_model.model().input
x=kg_model.model().layers[-1].output
x=Dense(2048, activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(2048, activation='relu')(x)
x=Dropout(0.5)(x)
outputs=Dense(len(tags), activation='sigmoid')(x)
kg_model._model=Model(inputs=inputs,outputs=outputs)
kg_model.compile()
kg_model.model().summary()



#
# MODEL-RUN
#
def run_model(n,id='A'):
    for i in range(n):
        run_name=f'{MODEL_NAME}_{id}_{i}'
        if i: epochs=20
        else: epochs=10
        print(f'\n\n{run_name}({i}):')
        kg_model.fit_gen(
             epochs=epochs,
             train_gen=train_gen,
             train_steps=200,
             validation_gen=valid_gen,
             validation_steps=100,
             history_name=run_name,
             checkpoint_name=run_name)


run_model(3)



