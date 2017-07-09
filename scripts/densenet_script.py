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
import helpers.dfgen as gen; reload(gen)
import models.resnet as resnet; reload(resnet)


DATA_ROOT=os.environ.get('DATA')
DATA_DIR='planet'
WEIGHT_ROOT=os.environ.get('WEIGHTS')
IMG_TYPE='tif'
ROOT=f'{DATA_ROOT}/{DATA_DIR}'
WEIGHT_DIR=f'{WEIGHT_ROOT}/{DATA_DIR}' # Weights are saved and loaded here
JPG_DIR = os.path.join(ROOT, 'train-jpg')
TIF_DIR = os.path.join(ROOT, 'train-tif')


TAGS=['primary','clear','agriculture','road','water','partly_cloudy','cultivation','habitation','haze','cloudy','bare_ground','selective_logging','artisinal_mine','blooming','slash_burn','conventional_mine','blow_down']

PRIMARY_LABELS=['primary'] 
AGRICULTURE_LABELS=['agriculture'] 
ROAD_LABELS=['road'] 
WATER_LABELS=['water'] 
CULTIVATION_LABELS=['cultivation'] 
HABITATION_LABELS=['habitation'] 
BAREGROUND_LABELS=['bare_ground'] 

WEATHER_LABELS=['clear','partly_cloudy','haze','cloudy'] 
RARE_LABELS=['selective_logging', 'artisinal_mine', 'blooming', 'slash_burn', 'conventional_mine', 'blow_down']

pld40_weather=helpers.PlanetData(train_size=40,tags=WEATHER_LABELS,create=True)
pld200_weather=helpers.PlanetData(train_size=200,tags=WEATHER_LABELS,create=True)
pld2000_weather=helpers.PlanetData(train_size=2000,tags=WEATHER_LABELS,create=True)
pldALL_weather=helpers.PlanetData(train_size='ALL',tags=WEATHER_LABELS,create=True)

densenet_weather=densenet.DenseNet(loss_func='categorical_crossentropy',
                               target_dim=len(WEATHER_LABELS)+1,
                               metrics=['accuracy'],
                               output_activation='softmax',image_ext='jpg')

#densenet_weather.fit_gen(batch_size=32,epochs=200,pdata=pldALL_weather,history_name='pldALL_densenet_jpg_weather')
densenet_weather.fit_gen(batch_size=2,epochs=2,pdata=pld40_weather)

# os.system("sudo poweroff")

