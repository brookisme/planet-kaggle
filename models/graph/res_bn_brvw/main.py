import os
import numpy as np
from dfgen import DFGen
from kgraph.functional import RESNET
"""
    INTERNAL
"""
#
# SETUP
#
PKR=os.environ['PKR']
TRAIN_CSV=f'{PKR}/datacsvs/train.csv'
VALID_CSV=f'{PKR}/datacsvs/valid.csv'

BATCH_SHAPE=(None,256,256,4)
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



"""
    PUBLIC
"""
#
# MODEL
#
BASE_WEIGHTS='tdn-brvw-adam-3.hdf5'
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
        'output': { 
                'units': 17,
                'activation': 'sigmoid' 
        }
}

def get_kg_model():
    return RESNET(graph)


#
# GENERATORS
#
TRAIN_BATCH_SIZE=32
VALID_BATCH_SIZE=16

train_gen=DFGen(
    csv_file=TRAIN_CSV,
    csv_sep=',',
    batch_size=TRAIN_BATCH_SIZE,
    lambda_func=norm_brvw)

valid_gen=DFGen(
    csv_file=VALID_CSV,
    csv_sep=',',
    batch_size=VALID_BATCH_SIZE,
    lambda_func=norm_brvw)

