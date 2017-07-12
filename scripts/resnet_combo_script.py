import sys

# REPO_PATH is the local git directory
REPO_PATH='/Users/Halmagyi/Documents/Programming/Python/ML/Kaggle/Planet/planet-kaggle'
sys.path.append(REPO_PATH)
REPO_PATH_GPU81='/home/Halmagyi/kaggle/planet/planet-kaggle'
sys.path.append(REPO_PATH_GPU81)

import os
import numpy as np
import pandas as pd
from importlib import reload
import utils
from keras import backend as K
from keras import metrics
from skimage import io
import pickle

import matplotlib.pyplot as plt

# Here we import the py files for current project.
import utils; reload(utils)
# import models.vgg16 as vl; reload(vl)
import models.ekami as ek; reload(ek)
import models.aframe as aframe; reload(aframe)
import models.base as base; reload(base)
import helpers.planet as helpers; reload(helpers)
from dfgen import DFGen
import models.resnet as resnet; reload(resnet)


DATA_ROOT=os.environ.get('DATA')
DATA_DIR='planet'
WEIGHT_ROOT=os.environ.get('WEIGHTS')
IMG_TYPE='tif'
ROOT=f'{DATA_ROOT}/{DATA_DIR}'
WEIGHT_DIR=f'{WEIGHT_ROOT}/{DATA_DIR}' # Weights are saved and loaded here
JPG_DIR = os.path.join(ROOT, 'train-jpg')
TIF_DIR = os.path.join(ROOT, 'train-tif')

TAGS=[
    'primary',
    'clear',
    'agriculture',
    'road',
    'water',
    'partly_cloudy',
    'cultivation',
    'habitation',
    'haze',
    'cloudy',
    'bare_ground',
    'selective_logging',
    'artisinal_mine',
    'blooming',
    'slash_burn',
    'conventional_mine',
    'blow_down']

PRIMARY_LABELS=['primary'] 
AGRICULTURE_LABELS=['agriculture'] 
ROAD_LABELS=['road'] 
WATER_LABELS=['water'] 
CULTIVATION_LABELS=['cultivation'] 
HABITATION_LABELS=['habitation'] 
BAREGROUND_LABELS=['bare_ground'] 
WEATHER_LABELS=['clear','partly_cloudy','haze','cloudy'] 
RARE_LABELS=['selective_logging','artisinal_mine','blooming','slash_burn','conventional_mine','blow_down']


image_types=['tif']*9
tag_types=[WEATHER_LABELS,RARE_LABELS,PRIMARY_LABELS,AGRICULTURE_LABELS,ROAD_LABELS,WATER_LABELS,CULTIVATION_LABELS,HABITATION_LABELS,BAREGROUND_LABELS]
labels_types=['weather','rare','primary','agriculture','road','water','cultivation','habitation','bareground']


BATCH_SIZE=32

train_gens=[]
valid_gens=[]
for tag_type in tag_types:
    tr_gen=DFGen(csv_file=f'{REPO_PATH_GPU81}/datacsvs/train.csv',csv_sep=',',batch_size=BATCH_SIZE)
    val_gen=DFGen(csv_file=f'{REPO_PATH_GPU81}/datacsvs/valid.csv',csv_sep=',',batch_size=BATCH_SIZE)
    tr_gen.reduce_columns(*tag_type,others=False)
    val_gen.reduce_columns(*tag_type,others=False)
    train_gens.append(tr_gen)
    valid_gens.append(val_gen)


resnet_models=[resnet.ResNet50(loss_func='categorical_crossentropy',
                               target_dim=len(tag_types[i]),
                               optimizer='sgd',
                               metrics=['accuracy'],
                               output_activation='softmax',image_ext=image_types[i]) for i in range(len(image_types))]
train_sz=train_gens[0].size
valid_sz=valid_gens[0].size


for i in range(len(tag_types)):
    resnet_models[i].fit_gen(batch_size=BATCH_SIZE,epochs=100,steps_per_epoch=30,
                       train_gen=train_gens[i],valid_gen=valid_gens[i],train_sz=train_sz,valid_sz=valid_sz,
                       history_name=f'{labels_types[i]}',reduce_lr=True)
    resnet_models[i].model().save_weights(f'{WEIGHT_ROOT}/resnet_{labels_types[i]}.hdf5')




os.system("sudo poweroff")
