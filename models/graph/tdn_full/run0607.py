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

RUN_NAME='full_0706'
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
tdn2.model().summary()



#
# RUN MODEL
#
