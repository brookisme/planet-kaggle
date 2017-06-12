

"""[markdown]
# GPU-Services (84)
"""


"""[code]"""
import os,sys
sys.path.append(os.environ.get('PKR'))
""""""


"""[code]"""
import pickle
import matplotlib.pyplot as plt
import utils
%matplotlib inline
""""""


"""[code]"""
HIST_DIR='../out/history'
""""""


"""[code]"""
def gethist(path):
    with open(f'{HIST_DIR}/{path}','rb') as file:
        hist=pickle.load(file)
    return hist

def plthist(file,xllim=None):
    utils.plot_dict(gethist(file),title=file,xllim=xllim)

def plthists(idents,pre=False,epochs=False,xllim=None):
    if epochs: tmpl='cl-{}.epoch.p'
    else: tmpl='cl-{}.batch.p'
    if pre: tmpl=f'pre-{tmpl}'
    paths=[tmpl.format(ident) for ident in idents]
    for path in paths:
        plthist(path,xllim)
""""""


"""[markdown]
### (32,[3,12]),(64,[3]),(16,[3])
"""


"""[code]"""
epath='cl-32_3.12-64_3-16_3.epoch.p'
hist=gethist(epath)
""""""


"""[code]"""
# hist
""""""


"""[raw]
{'acc': 0.91047795414924626,
 'k_f2': 0.62089061737060547,
 'loss': 0.23782273977994919,
 'val_acc': 0.91819855570793152,
 'val_k_f2': 0.63012641668319702,
 'val_loss': 0.23725360631942749}
"""


"""[markdown]
## ((32,[3]),(64,[3]),(128,[3])
"""


"""[code]"""
idents=['32_3-64_3-128_3','32_3.3-64_3.3-128_3.3.3']
""""""


"""[code]"""
plthists(idents,pre=True,xllim=325)
""""""


"""[code]"""
plthists(idents,pre=False,xllim=325)
""""""






