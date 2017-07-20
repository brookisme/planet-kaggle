import os
import main
from dfgen import DFGen
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.models import load_model
import kgraph.functional as fnc



MODEL_NAME='combo'
TRAIN_BATCH_SIZE=64 #main.TRAIN_BATCH_SIZE
main.train_gen.batch_size=TRAIN_BATCH_SIZE
VALID_BATCH_SIZE=main.VALID_BATCH_SIZE
TRAIN_STEPS=200
VALID_STEPS=100
RUN_START=0



def save_model(name):
    kg_model.model().save(f'models/{name}.hdf5')


def run_model(n,ident='A'):
    for i in range(RUN_START,RUN_START+n):
        run_name=f'{MODEL_NAME}_{ident}_{i}'
        if (RUN_START-i): epochs=10
        else: epochs=10
        print(f'\n\n{run_name}:')
        kg_model.fit_gen(
             epochs=epochs,
             train_gen=main.train_gen,
             train_steps=TRAIN_STEPS,
             validation_gen=main.valid_gen,
             validation_steps=VALID_STEPS,
             history_name=run_name,
             checkpoint_name=run_name)
        save_model(run_name)


def load_models(model_names):
    return [load_model(f'models/{name}') for name in model_names]


#
# EXISTING MODELS
#

model_names=[
    'primary_1024x512_0.hdf5',
    'agriculture_1024x512_1.hdf5',
    'road_1024x512_1.hdf5',
    'water_1024x512_1.hdf5',
    'cultivation_1024x512_0.hdf5',
    'habitation_1024x512_2.hdf5',
    'weather_1024x512_3.hdf5',
    'rare_1024x512_1.hdf5', 
]

models=load_models(model_names)


#
# COMBO MODEL
#

COMPLEXITY=[512,512]
print("\n\n\n\n\n{}".format(COMPLEXITY))

graph={
    'meta': {
        'network_type': 'Combo'
    },
    'compile': {
            'loss_func': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'adam'
    },
    'inputs': {
        'batch_shape':(None,256,256,4),
    },
    'fc_blocks':[
        { 'units': 512 },
        { 'units': 512 }
    ],
    'output': { 
        'units': 17,
        'activation': 'sigmoid' 
    }
}
kg_model=fnc.Combo(input_models=models,graph=graph)



# kg_model.model().summary()



kg_model.compile()




run_model(6,"x".join(map(str,COMPLEXITY)) or 'linearmap')

#
# MODEL-RUN
#




del kg_model

COMPLEXITY=[1024,512]
print("\n\n\n\n\n{}".format(COMPLEXITY))

graph={
    'meta': {
        'network_type': 'Combo'
    },
    'compile': {
            'loss_func': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'adam'
    },
    'inputs': {
        'batch_shape':(None,256,256,4),
    },
    'fc_blocks':[
        { 'units': 1024 },
        { 'units': 512 }
    ],
    'output': { 
        'units': 17,
        'activation': 'sigmoid' 
    }
}
kg_model=fnc.Combo(input_models=models,graph=graph)



# kg_model.model().summary()



kg_model.compile()

run_model(6,"x".join(map(str,COMPLEXITY)) or 'linearmap')




del kg_model

COMPLEXITY=[256,256]
print("\n\n\n\n\n{}".format(COMPLEXITY))

graph={
    'meta': {
        'network_type': 'Combo'
    },
    'compile': {
            'loss_func': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'optimizer': 'adam'
    },
    'inputs': {
        'batch_shape':(None,256,256,4),
    },
    'fc_blocks':[
        { 'units': 256 },
        { 'units': 256 }
    ],
    'output': { 
        'units': 17,
        'activation': 'sigmoid' 
    }
}
kg_model=fnc.Combo(input_models=models,graph=graph)



# kg_model.model().summary()



kg_model.compile()

run_model(6,"x".join(map(str,COMPLEXITY)) or 'linearmap')




print("\n\n\n\n\n--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
print("--------------------------")
# os.system("poweroff")



