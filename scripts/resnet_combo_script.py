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
LABEL_SAMPLE_CSV=os.path.join(ROOT,'train_sample.csv')

image_types=['jpg']+['tif']*8
tag_types=[WEATHER_LABELS,RARE_LABELS,PRIMARY_LABELS,AGRICULTURE_LABELS,ROAD_LABELS,WATER_LABELS,CULTIVATION_LABELS,HABITATION_LABELS,BAREGROUND_LABELS]
labels_types=['weather','rare','primary','agriculture','road','water','cultivation','habitation','bareground']


BATCH_SIZE=32
gen=DFGen(csv_file=f'{ROOT}/train.csv',csv_sep=',',batch_size=BATCH_SIZE)
gen.save(path='combo_train.csv',split_path='combo_valid.csv')

train_gens=[DFGen(csv_file='combo_train.csv',csv_sep=',',batch_size=BATCH_SIZE)
            .reduce_columns(*tag_type,others=False) for tag_type in tag_types]

valid_gens=[DFGen(csv_file='combo_valid.csv',csv_sep=',',batch_size=BATCH_SIZE)
            .reduce_columns(*tag_type,others=False) for tag_type in tag_types]


resnet_models=[resnet.ResNet50(loss_func='categorical_crossentropy',
                               target_dim=len(tag_types[i])+1,
                               metrics=['accuracy'],
                               output_activation='softmax',image_ext=image_types[i]) for i in range(len(image_types))]



resnet_models[0].fit_gen(batch_size=BATCH_SIZE,epochs=100,steps_per_epoch=20,
                       train_gen=train_gen,valid_gen=valid_gen,train_sz=train_sz,valid_sz=valid_sz,
                       history_name='weather',reduce_lr=True)









os.system("sudo poweroff")

