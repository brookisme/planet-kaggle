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

RUN_NAME='full_graph_0706'


#
# SPLIT DATA
#
DATA=f'{os.environ["DATA"]}/planet'
CSV=f'{DATA}/train.csv'
gen=DFGen(csv_file=CSV,csv_sep=',')
gen.dataframe=gen.dataframe.drop(['labels','paths'],axis=1)
gen.save(path=CSV)
gen.save(path='train.csv',split_path='valid.csv',split=0.2)


#
# SETUP DATA
#
train_gen=DFGen(csv_file='train.csv',csv_sep=',')
valid_gen=DFGen(csv_file='valid.csv',csv_sep=',')
print(train_gen.size,valid_gen.size)
train_gen.dataframe.head(3)



#
# SETUP/CHECK MODEL
#
tdn=T(m.graph)
tdn.graph
tdn.model().summary()


#
# RUN MODEL
#
tdn.fit_gen(
   epochs=60,
   train_gen=train_gen,
   train_steps=200,
   validation_gen=valid_gen,
   validation_steps=100,
   history_name=RUN_NAME,
   checkpoint_name=RUN_NAME)