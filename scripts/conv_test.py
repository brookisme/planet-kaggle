import os,sys
sys.path.append(os.environ.get('PKR'))
import math
import numpy as np
from skimage import io
from keras import backend as K
from keras.optimizers import Adam 
import models.aframe as af
from helpers.planet import PlanetData
from helpers.dfgen import DFGen
import utils


# pld=PlanetData(create=True,train_size='FULL')
pld=PlanetData(create=True,train_size=500)
# pld=PlanetData(train_size=250)

#
# MODEL
#
conv_layers=None
fc_layers=[
    256,
    512]

def run(ident,conv_list,n,epochs,lr,sz_multi):
    #
    # setup model
    #
    conv_layers=[
        (32,conv_list),
        (64,[3]),
        (16,[3])]
    
    conv_ident=''.join(map(str,conv_list))

    flex=af.Flex(
        conv_layers=conv_layers,
        fc_layers=fc_layers,
        batch_norm=True,
        metrics=['accuracy',utils.k_f2])
    train_sz=sz_multi*32
    valid_sz=math.floor(train_sz*0.20)
    flex.compile(optimizer=Adam(lr=lr))
    #
    # setup model
    #
    for i in range(n):
        print(f'\n\n{ident}: {i}')
        flex.fit_gen(
            epochs=epochs,
            pdata=pld,
            batch_size=64)
        if (i%2==0):
            flex.save_weights(f'flex{conv_ident}_{ident}_{i}')



run('pre',[3,12],2,5,0.001,5)
# run('lr02',2,15,0.02,50)
# run('lr01',2,15,0.01,100)









