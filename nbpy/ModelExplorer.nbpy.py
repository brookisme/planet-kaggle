

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
# ek_bc=ek.EKAMI(metrics=['accuracy',utils.k_f2])
# ek_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""


"""[raw]
start with batch norm... (None, 256, 256, 4)
Epoch 1/2
125/125 [==============================] - 313s - loss: 0.2542 - acc: 0.9029 - k_f2: 0.6155 - val_loss: 0.2693 - val_acc: 0.8860 - val_k_f2: 0.6009
Epoch 2/2
125/125 [==============================] - 308s - loss: 0.1458 - acc: 0.9444 - k_f2: 0.7743 - val_loss: 0.2877 - val_acc: 0.8993 - val_k_f2: 0.6359
"""


"""[code]"""
### BEFORE FLEX FLEXIBILITY
# flex_bc=af.Flex(metrics=['accuracy',utils.k_f2])
# flex_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""


"""[raw]
Epoch 1/2
125/125 [==============================] - 308s - loss: 0.2525 - acc: 0.9043 - k_f2: 0.6189 - val_loss: 0.2537 - val_acc: 0.9085 - val_k_f2: 0.5933
Epoch 2/2
125/125 [==============================] - 308s - loss: 0.1733 - acc: 0.9334 - k_f2: 0.7196 - val_loss: 0.2538 - val_acc: 0.9163 - val_k_f2: 0.6580
"""


"""[code]"""
# flex_bc=af.Flex(metrics=['accuracy',utils.k_f2])
# flex_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""


"""[code]"""
af.Flex(metrics=['accuracy',utils.k_f2]).model().summary()
""""""


"""[raw]
Epoch 1/2
125/125 [==============================] - 309s - loss: 0.2547 - acc: 0.9024 - k_f2: 0.6154 - val_loss: 0.2300 - val_acc: 0.9076 - val_k_f2: 0.6551
Epoch 2/2
125/125 [==============================] - 309s - loss: 0.1810 - acc: 0.9297 - k_f2: 0.7066 - val_loss: 0.2046 - val_acc: 0.9301 - val_k_f2: 0.7066
"""


"""[markdown]
## FLEXY FLEX
"""


"""[code]"""
conv_layers=[
    (32,[3,6,12]),
    (16,[3,5])]

fc_layers=[
    256,
    512,
    128]
""""""


"""[code]"""
flex_bc_a=af.Flex(conv_layers=conv_layers,fc_layers=fc_layers,metrics=['accuracy',utils.k_f2])
""""""


"""[code]"""
flex_bc_a.model().summary()
""""""


"""[code]"""
flex_bc_a.fit_gen(epochs=2,pdata=pld,sample_pct=0.15,ndvi_images=False)
""""""


"""[code]"""
flex_bc_a.compile(optimizer=Adam(lr=0.01))
""""""


"""[code]"""
flex_bc_a.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""






"""[markdown]
## NO BATCH NORM - NO NDVI -  LR*10
"""


"""[code]"""
ek_bc_01=ek.EKAMI(metrics=['accuracy',utils.k_f2],optimizer=Adam(lr=0.01))
ek_bc_01.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""


"""[code]"""
flex_bc_01=af.Flex(metrics=['accuracy',utils.k_f2],optimizer=Adam(lr=0.01))
flex_bc_01.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""




