

"""[markdown]
## Testing Playground Noise
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
from activations.stepfunc import Stepfunc
""""""


"""[code]"""
import matplotlib.pyplot as plt
%matplotlib inline
""""""


"""[code]"""
from skimage import io
""""""


"""[code]"""
from keras.layers import Dense
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


"""[code]"""
ek_bc=ek.EKAMI(loss_func='binary_crossentropy')
""""""


"""[code]"""
for layer in ek_bc.model().layers:
    print(layer.trainable)
""""""


"""[code]"""
ek_bc.model().summary()
""""""


"""[code]"""
ek_bc.load_weights('pretrain-steppy-test')
""""""


"""[code]"""
for layer in ek_bc.model().layers:
    layer.trainable=False
""""""


"""[code]"""
model=ek_bc.model()
""""""


"""[code]"""
for layer in model.layers:
    print(layer.trainable)
""""""


"""[code]"""
model.add(Dense(17))
model.add(Stepfunc())
""""""


"""[code]"""
for layer in model.layers:
    print(layer.trainable)
""""""


"""[code]"""
model.summary()
""""""


"""[markdown]
### fig-gen to function
"""


"""[code]"""

def fit_gen(
       model,
       epochs=None,
       pdata=None,
       sample_pct=1.0,
       batch_size=32):
   """ call fit_generator
       Args:
           -   if pdata (instance of <data.planent:PlanetData>)
               use params from pdata
           -   otherwise used passed params
   """
   if pdata:
       train_sz=pdata.train_size
       valid_sz=pdata.valid_size
       train_gen=gen.DFGen(
           dataframe=pdata.train_df,batch_size=batch_size).data()
       valid_gen=gen.DFGen(
           dataframe=pdata.valid_df,batch_size=batch_size).data()

   nb_epochs,steps,validation_steps=utils.gen_params(
       train_sz,valid_sz,epochs,sample_pct)
   return model.fit_generator(
       generator=train_gen,
       validation_data=valid_gen,
       steps_per_epoch=steps,
       validation_steps=validation_steps,
       epochs=epochs,
       verbose=1)
""""""


"""[code]"""
pld=data.PlanetData(train_size=300)
""""""


"""[code]"""
fit_gen(model,epochs=2,pdata=pld,sample_pct=1)
""""""


"""[code]"""
PROJECT_NAME='planet'
DATA_ROOT=os.environ.get('DATA')
WEIGHT_ROOT=os.environ.get('WEIGHTS')
DATA_DIR=f'{DATA_ROOT}/{PROJECT_NAME}'
WEIGHT_DIR=f'{WEIGHT_ROOT}/{PROJECT_NAME}'
""""""


"""[code]"""
print(WEIGHT_DIR)
""""""


"""[code]"""
model.save_weights(f'{WEIGHT_DIR}/howinthe.1.hdf5')
""""""


"""[code]"""
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy',utils.k_f2])
""""""


"""[code]"""
fit_gen(model,epochs=2,pdata=pld,sample_pct=1)
""""""


"""[code]"""
model.save_weights(f'{WEIGHT_DIR}/howinthe.2.hdf5')
""""""


"""[markdown]
## PREDICITONS
"""


"""[code]"""
ek_bc=ek.EKAMI(loss_func='binary_crossentropy')
ek_bc.load_weights('pretrain-steppy-test')
""""""


"""[code]"""
pld=data.PlanetData(train_size=300)
""""""


"""[code]"""
#
# Prediction for random image in dataframe df
#
def predict_random(model_obj,df,file_ext='tif',image_dir='train-tif',noisy=True,return_image=False):
    imdf=df.sample()
    pred,img=model_obj.predict_image(
        imdf.image_name.values[0],
        file_ext='tif',
        image_dir='train-tif',
        return_image=True)
    im_name=imdf.image_name.values[0]
    vec=imdf.vec.values[0]
    prediction=[int(round(i)) for i in pred[0]]
    eq=np.array_equal(vec,prediction)
    loss=utils.cos_distance(vec,prediction,return_type='float')
    loss="%.3f" % round(loss,3)
    if noisy:
        print('\nimage:',im_name)
        print('vec:',vec)
        print('prd:',prediction)
        print('equal:',eq)
        print('dist:',loss)
    if return_image:
        return loss, eq, im_name, img
    else:
        return vec, prediction

""""""


"""[markdown]
#### validation f2
"""


"""[code]"""
trues,preds=[],[]
for i in range(100):
    t,p=predict_random(ek_bc,pld.valid_df,noisy=False)
    trues.append(t)
    preds.append(p)
""""""


"""[code]"""
utils.f2_score(trues,preds)
""""""


"""[markdown]
#### train f2
"""


"""[code]"""
trues,preds=[],[]
for i in range(100):
    t,p=predict_random(ek_bc,pld.train_df,noisy=False)
    trues.append(t)
    preds.append(p)
""""""


"""[code]"""
utils.f2_score(trues,preds)
""""""


"""[code]"""
model=ek_bc.model()
""""""


"""[code]"""
for layer in model.layers:
    layer.trainable=False
""""""


"""[code]"""
model.add(Dense(17))
model.add(Stepfunc())
""""""


"""[code]"""
model.summary()
""""""


"""[code]"""
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy',utils.k_f2])
""""""


"""[code]"""
fit_gen(model,epochs=2,pdata=pld,sample_pct=1)
""""""


"""[code]"""
#
# Prediction for random image in dataframe df
#
def _image_path(name=None,file_ext=None,data_root=DATA_DIR,image_dir=None):
    fpath=f'{data_root}'
    if image_dir: fpath=f'{fpath}/{image_dir}'
    fpath=f'{fpath}/{name}'
    if file_ext: fpath=f'{fpath}.{file_ext}'
    return fpath

def pr(model,df,file_ext='tif',image_dir='train-tif',noisy=True,return_image=False):
    imdf=df.sample()
    im_name=imdf.image_name.values[0]
    image=io.imread(_image_path(im_name,'tif',DATA_DIR,'train-tif'))
    pred=model.predict(np.expand_dims(image, axis=0))
    vec=imdf.vec.values[0]
    prediction=[int(round(i)) for i in pred[0]]
    eq=np.array_equal(vec,prediction)
    loss=utils.cos_distance(vec,prediction,return_type='float')
    loss="%.3f" % round(loss,3)
    if noisy:
        print('\nimage:',im_name)
        print('vec:',vec)
        print('prd:',prediction)
        print('equal:',eq)
        print('dist:',loss)
    if return_image:
        return loss, eq, im_name, img
    else:
        return vec, prediction


""""""


"""[code]"""
pr(model,pld.train_df)
""""""


"""[code]"""
pr(model,pld.train_df)
""""""


"""[code]"""
pr(model,pld.train_df)
""""""


"""[code]"""
pr(model,pld.train_df)
""""""


"""[code]"""
pr(model,pld.train_df)
""""""


"""[code]"""
pr(model,pld.train_df)
""""""


