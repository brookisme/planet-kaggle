

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
""""""


"""[code]"""
import models.vgglike as vl
import models.ekami as ek
import helpers.planet as data
import helpers.dfgen as gen
# import data.planet as data
# import data.dfgen as gen

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
## numpy ndvi testing
"""


"""[code]"""
r=np.array([
    [1.0,1.0],
    [1.0,1.0]])

g=np.array([
    [2.0,2.0],
    [2.0,2.0]])

b=np.array([
    [3.0,3.0],
    [3.0,3.0]])

ir=np.array([
    [4.0,4.0],
    [4.0,4.0]])

""""""


"""[code]"""
# desired output
imgood=np.array([
    [[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]],
    [[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]]])
print(imgood.shape)
imgood
""""""


"""[code]"""
img=np.array([r,g,b,ir])
print(img.shape)
img
""""""


"""[code]"""
bands=[bnd.reshape(2,2,1) for bnd in [r,g,b,ir]]
""""""


"""[code]"""
bands[0].shape
""""""


"""[code]"""
img=np.concatenate(bands,axis=2)
""""""


"""[code]"""
img.shape
""""""


"""[code]"""
img
""""""


"""[code]"""
img[:,:,3]
""""""


"""[code]"""
rimg=img[:,:,0]
irimg=img[:,:,3]
ndvi=(irimg-rimg)/(irimg+rimg)
""""""


"""[code]"""
ndvi
""""""


"""[code]"""
(4-1)/(4+1)
""""""


"""[markdown]
## lambda ndvi
"""


"""[code]"""
def ndvi(img):
    r=img[:,:,0]
    nir=img[:,:,3]
    return (nir-r)/(nir+r)


def ndviimg(img):
    img=img.copy()
    ndvi_band=ndvi(img)
    img[:,:,3]=ndvi_band
    return img
""""""


"""[code]"""
img
""""""


"""[code]"""
ndvi(img)
""""""


"""[code]"""
ndviimg(img)
""""""


"""[markdown]
## TENSOR
"""


"""[code]"""
img
""""""


"""[code]"""
t=K.variable(img)
""""""


"""[code]"""
print(t,K.eval(t))
""""""


"""[code]"""
K.eval(ndvi(t))
""""""


"""[code]"""
K.eval(ndviimg(t))
""""""


"""[code]"""
def ndvit(img):
    ndvi_band=ndvi(img)
    img[:,:,3]=ndvi_band
    return img
""""""


"""[code]"""
K.eval(ndvit(t))
""""""


"""[code]"""
rgb=t[:,:,:3]
""""""


"""[code]"""
K.eval(rgb)
""""""


"""[code]"""
n=ndvi(t);print(n.shape); K.eval(n)
""""""


"""[code]"""
nv=K.reshape(n,(2,2,1))
""""""


"""[code]"""
nd=K.concatenate((rgb,nv),-1)
""""""


"""[code]"""
K.eval(nd)
""""""


"""[code]"""
def ndvit(img,sz=256):
    t=K.placeholder((sz,sz,4), dtype='float32')
    if not K.is_keras_tensor(t):
        K.update(t,K.variable(img))
    else:
        t=img
    t=K.cast(t,dtype='float32')
    print("get rgb")
    rgb=t[:,:,:3]
    print("\n\n",rgb)
    print("get ndvi")
    ndvi_band=K.reshape(ndvi(t),(sz,sz,1))
    print("\n\n",K.eval(ndvi_band))
    print("concat")
    return K.concatenate((rgb,ndvi_band),-1)
""""""


"""[code]"""
img
""""""


"""[code]"""
K.eval(ndvit(K.variable(img),2))
""""""


"""[markdown]
## GO
"""


"""[code]"""
# pld=data.PlanetData(create=True,train_size=100)
pld=data.PlanetData(train_size=300)
""""""


"""[code]"""
ek_lmbd_bc=ek.EKAMI(loss_func='binary_crossentropy')
""""""


"""[code]"""
ek_lmbd_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=True)
""""""


"""[code]"""
ek_no_bc=ek.EKAMI(loss_func='binary_crossentropy')
ek_no_bc.fit_gen(epochs=2,pdata=pld,sample_pct=1,ndvi_images=False)
""""""


"""[markdown]
## debug
"""


"""[code]"""
PROJECT_NAME='planet'
DATA_ROOT=os.environ.get('DATA')
WEIGHT_ROOT=os.environ.get('WEIGHTS')
DATA_DIR=f'{DATA_ROOT}/{PROJECT_NAME}'
WEIGHT_DIR=f'{WEIGHT_ROOT}/{PROJECT_NAME}'
""""""


"""[code]"""
def _image_path(name=None,file_ext=None,data_root=DATA_DIR,image_dir=None):
    fpath=f'{data_root}'
    if image_dir: fpath=f'{fpath}/{image_dir}'
    fpath=f'{fpath}/{name}'
    if file_ext: fpath=f'{fpath}.{file_ext}'
    return fpath
""""""


"""[code]"""
def randimg(df,noisy=False):
    rdf=df.sample()
    if noisy: print(rdf)
    path=_image_path(rdf.image_name.values[0],'tif',DATA_DIR,'train-tif')
    return io.imread(path)
""""""


"""[code]"""
im=randimg(pld.train_df)
print(im)
(3869-4215)/(3869+4215)
""""""


"""[code]"""
ndvi=nir2ndvi(im)
""""""


"""[code]"""
im.shape
""""""


"""[code]"""
ndvi
""""""


