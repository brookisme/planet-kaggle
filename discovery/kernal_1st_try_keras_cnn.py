# First Try using Keras + CNN
import os
import numpy as np
import pandas as pd
from glob import glob

DATA_ROOT=os.environ.get('DATA')
DATA_DIR='planet/sample'
DATA=f'{DATA_ROOT}/{DATA_DIR}'
TARGET=f'{DATA}/train.csv'
TRAIN_JPG=f'{DATA}/train-jpg'
TRAIN_TIF=f'{DATA}/train-tif'
TEST_JPG=f'{DATA}/test-jpg'
TEST_TIF=f'{DATA}/test-tif'


def hello_repo():
    df_target=pd.read_csv(TARGET)
    return df_target.head()