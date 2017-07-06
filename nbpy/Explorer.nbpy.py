

"""[code]"""
import keras
keras.__version__
""""""


"""[code]"""
import os
from importlib import reload
from pprint import pprint
import dfgen as dfg
import kgraph as kg
import model as m
reload(dfg)
reload(kg)
reload(m)
from dfgen import DFGen
from kgraph.functional import TIRAMISU as T
""""""


"""[markdown]
### SETUP
"""


"""[code]"""
DATA=f'{os.environ["DATA"]}/planet'
""""""


"""[code]"""
CSV=f'{DATA}/train.csv'
""""""


"""[code]"""
pprint(m.graph)
""""""


"""[markdown]
### DEV DATA SETUP
"""


"""[code]"""
gen=DFGen(csv_file=CSV,csv_sep=',')
""""""


"""[code]"""
gen.dataframe.sample(3)
""""""


"""[code]"""
# print('full:',gen.size)
# gen.limit(8000)
# print('limited:',gen.size)
# gen.save(path='large_dev_train.csv',split_path='large_dev_valid.csv',split=0.2)
""""""


"""[code]"""
train_gen=DFGen(csv_file='large_dev_train.csv',csv_sep=',')
valid_gen=DFGen(csv_file='large_dev_valid.csv',csv_sep=',',batch_size=32)
""""""


"""[code]"""
print(train_gen.size)
train_gen.dataframe.sample(3)
""""""


"""[code]"""
print(valid_gen.size)
valid_gen.dataframe.sample(3)
""""""


"""[code]"""
image_batch=next(train_gen)[0]
print(image_batch.shape)
label_batch=next(train_gen)[1]
print(label_batch.shape,label_batch[0])
""""""


"""[code]"""
# train_gen.limit(300)
# valid_gen.limit(100)
# train_gen.save(path='dev_train.csv')
# valid_gen.save(path='dev_valid.csv')
train_gen=None
valid_gen=None
""""""


"""[markdown]
### MODEL
"""


"""[code]"""
tdn=T(m.graph)
""""""


"""[code]"""
tdn.graph
""""""


"""[code]"""
tdn.model().summary()
""""""


"""[markdown]
### TRAIN
"""


"""[code]"""
train_gen=DFGen(csv_file='dev_train.csv',csv_sep=',')
valid_gen=DFGen(csv_file='dev_valid.csv',csv_sep=',',batch_size=32)
""""""


"""[code]"""
print(train_gen.size,valid_gen.size)
train_gen.dataframe.sample(3)
""""""


"""[code]"""
tdn.fit_gen(
    epochs=1,
    train_gen=train_gen,
    train_steps=10,
    history_name='dev_run_1',
    checkpoint_name='dev_run_1')
""""""


"""[code]"""
tdn.fit_gen(
            epochs=5,
            train_gen=train_gen,
            train_steps=100,
            validation_gen=valid_gen,
            validation_steps=50,
            history_name='dev_run_a',
            checkpoint_name='dev_run_a')
""""""


