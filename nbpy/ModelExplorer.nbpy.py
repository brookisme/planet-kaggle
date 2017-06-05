

"""[code]"""
import sys
sys.path.append('..')
""""""


"""[code]"""
import os
import numpy as np
from importlib import reload
import utils
""""""


"""[code]"""
from skimage import io
from keras import backend as K
from keras.optimizers import Adam
""""""


"""[code]"""
import models.vgglike as vl
import models.ekami as ek
import models.aframe as af
import helpers.planet as data
import helpers.dfgen as gen
import activations.stepfunc as sf
reload(data)
reload(gen)
reload(vl)
reload(utils)
reload(data)
reload(gen)
reload(ek)
reload(utils)
reload(sf)
""""""


"""[code]"""
# pld=data.PlanetData(create=True,train_size=250)
pld=data.PlanetData(train_size=250)
""""""


"""[markdown]
## NO BATCH NORM - NO NDVI - DEFAULT LR
"""


"""[code]"""
ek_bc=ek.EKAMI(metrics=['accuracy',utils.k_f2])
ek_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""


"""[code]"""
flex_bc=af.Flex(metrics=['accuracy',utils.k_f2])
flex_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""


"""[markdown]
## NO BATCH NORM - NO NDVI -  LR
"""


"""[code]"""
ek_bc_01=ek.EKAMI(metrics=['accuracy',utils.k_f2],optimizer=Adam(lr=0.01))
ek_bc_01.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""


"""[code]"""
flex_bc_01=af.Flex(metrics=['accuracy',utils.k_f2],optimizer=Adam(lr=0.01))
flex_bc_01.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""
