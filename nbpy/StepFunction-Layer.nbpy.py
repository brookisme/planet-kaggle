

"""[markdown]
In this notebook I am going to be investigating using and additional layer with a step function activation.
My first shot at the step function is K.round. I am thinking it doesn't actually matter where we threshold, since the the weights should take care of that.
"""


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
import matplotlib.pyplot as plt
%matplotlib inline
""""""


"""[code]"""
from skimage import io
""""""


"""[code]"""
from keras import backend as K
""""""


"""[code]"""
import models.vgglike as vl
import models.ekami as ek
import data.planet as data
import data.dfgen as gen
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


"""[markdown]
## DATA
"""


"""[code]"""
# pld=data.PlanetData(create=True,train_size=300)
pld=data.PlanetData(train_size=300)
""""""


"""[markdown]
## EKPLUS
"""


"""[code]"""
ekp7_bc=ek.EKPLUS(loss_func='binary_crossentropy')
ekp7_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1)
""""""


"""[code]"""
ekp6_bc=ek.EKPLUS(loss_func='binary_crossentropy')
ekp6_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1)
""""""


"""[raw]
N=20: loss: 0.4899 - acc: 0.8958 - val_loss: 0.5094 - val_acc: 0.9060
"""


"""[code]"""
ekp5_bc=ek.EKPLUS(loss_func='binary_crossentropy')
ekp5_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1)
""""""


"""[code]"""
ekp4_bc=ek.EKPLUS(loss_func='binary_crossentropy')
ekp4_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1)
""""""


"""[code]"""
ekp_bc=ek.EKPLUS(loss_func='binary_crossentropy')
""""""


"""[code]"""
ekp_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1)
""""""


"""[raw]
N=6: 372s - loss: 0.1747 - acc: 0.9321 - val_loss: 0.2525 - val_acc: 0.9106
"""


"""[code]"""
ekp2_bc=ek.EKPLUS(loss_func='binary_crossentropy')
""""""


"""[code]"""
ekp2_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1)
""""""


"""[raw]
N=1: 371s - loss: 0.2962 - acc: 0.8959 - val_loss: 0.2687 - val_acc: 0.9099
"""


"""[code]"""
ekp3_bc=ek.EKPLUS(loss_func='binary_crossentropy')
""""""


"""[code]"""
ekp3_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1)
""""""


"""[raw]
N=4: 373s - loss: 0.2214 - acc: 0.9130 - val_loss: 0.2234 - val_acc: 0.9162
"""


"""[markdown]
## EKAMI
"""


"""[code]"""
ek_bc=ek.EKAMI(loss_func='binary_crossentropy')
""""""


"""[code]"""
ek_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1)
""""""


"""[raw]
EKAMI: 379s - loss: 0.1818 - acc: 0.9289 - val_loss: 0.2474 - val_acc: 0.9146
"""


"""[markdown]
#### save weights and then add steppy
"""


"""[code]"""
ek_bc.save_weights('pretrain-steppy-test')
""""""


