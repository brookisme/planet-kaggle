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

WEATHER_LABELS=['clear','partly_cloudy','haze','cloudy'] 
RARE_LABELS=['selective_logging', 'artisinal_mine', 'blooming', 'slash_burn', 'conventional_mine', 'blow_down']

pld40_weather=helpers.PlanetData(train_size=40,tags=WEATHER_LABELS,create=True)
pld200_weather=helpers.PlanetData(train_size=200,tags=WEATHER_LABELS,create=True)
pld2000_weather=helpers.PlanetData(train_size=2000,tags=WEATHER_LABELS,create=True)
pldALL_weather=helpers.PlanetData(train_size='ALL',tags=WEATHER_LABELS,create=True)

def first_ele(ll):
    gen=(i for i,e in enumerate(ll) if e==1)
    idlist=[0]*len(ll)
    idlist[next(gen)]=1
    return idlist

def first_ele_pdata(pdata):
    pdata.train_df['vec']=pdata.train_df['vec'].apply(lambda x: first_ele(x))
    pdata.valid_df['vec']=pdata.valid_df['vec'].apply(lambda x: first_ele(x))
    return pdata

pld40_rare=first_ele_pdata(helpers.PlanetData(train_size=40,tags=RARE_LABELS,create=True))
pld200_rare=first_ele_pdata(helpers.PlanetData(train_size=200,tags=RARE_LABELS,create=True))
pld2000_rare=first_ele_pdata(helpers.PlanetData(train_size=2000,tags=RARE_LABELS,create=True))
pldALL_rare=first_ele_pdata(helpers.PlanetData(train_size='ALL',tags=RARE_LABELS,create=True))

# pld40_weather=helpers.PlanetData(train_size=40,tags=WEATHER_LABELS,create=False)
# pld200_weather=helpers.PlanetData(train_size=200,tags=WEATHER_LABELS,create=False)
# pld2000_weather=helpers.PlanetData(train_size=2000,tags=WEATHER_LABELS,create=False)
# pldALL_weather=helpers.PlanetData(train_size='ALL',tags=WEATHER_LABELS,create=False)

# pld40_rare=first_ele_pdata(helpers.PlanetData(train_size=40,tags=RARE_LABELS,create=False))
# pld200_rare=first_ele_pdata(helpers.PlanetData(train_size=200,tags=RARE_LABELS,create=False))
# pld2000_rare=first_ele_pdata(helpers.PlanetData(train_size=2000,tags=RARE_LABELS,create=False))
# pldALL_rare=first_ele_pdata(helpers.PlanetData(train_size='ALL',tags=RARE_LABELS,create=False))



resnet_weather=resnet.ResNet50(loss_func='categorical_crossentropy',
                               target_dim=4,
                               metrics=['accuracy'],
                               output_activation='softmax',image_ext='jpg')

resnet_rare=resnet.ResNet50(loss_func='categorical_crossentropy',
                               target_dim=len(RARE_LABELS),
                               metrics=['accuracy'],
                               output_activation='softmax',image_ext='tif')

#resnet_weather.fit_gen(batch_size=32,epochs=200,pdata=pldALL_weather,history_name='pldALL_resnet_jpg_weather')
#resnet_weather.fit_gen(batch_size=2,epochs=2,pdata=pld40_weather)

resnet_rare.fit_gen(batch_size=128,epochs=10,pdata=pldALL_rare,history_name='pldALL_resnet_tif_rare')
#resnet_rare.fit_gen(batch_size=2,epochs=4,pdata=pld40_rare)


os.system("sudo poweroff")

