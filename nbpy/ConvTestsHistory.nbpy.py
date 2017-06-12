

"""[markdown]
# GPU-Services (84)
"""


"""[code]"""
import os,sys
sys.path.append('PKR')
import pickle
""""""


"""[code]"""
HIST_DIR='../out/history'
""""""


"""[code]"""
def gethist(path):
    with open(f'{HIST_DIR}/{path}','rb') as file:
        hist=pickle.load(file)
    return hist
""""""


"""[markdown]
### (32,[3,12]),(64,[3]),(16,[3])
"""


"""[code]"""
epath='cl-32_3.12-64_3-16_3.epoch.p'
hist=gethist(epath)
""""""


"""[code]"""
hist
""""""


