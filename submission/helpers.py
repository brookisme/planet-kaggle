import os
import numpy as np
import processors as p
import math
from keras.models import load_model



BATCHSIZE=64
REPO_DIR=f'{os.environ["PKR"]}'
DATA_DIR=f'{os.environ["DATA"]}/planet'
TEST_DIR=f'{DATA_DIR}/test-tif'
TRAIN_DIR=f'{DATA_DIR}/train-tif'
TRAIN_CSV=f'{REPO_DIR}/datacsvs/train.csv'
VALID_CSV=f'{REPO_DIR}/datacsvs/valid.csv'



#
# NORMILAZATION
#
BATCH_SHAPE=(None,256,256,4)
R_MEAN=3072.1367933921815
R_STDEV=450.97146444273375

G_MEAN=4268.506753740692
G_STDEV=409.536961252693

B_MEAN=4969.024985050201
B_STDEV=394.0761093545407

N_MEAN=6400.041219993591
N_STDEV=856.3157545106753

V_MEAN=1.0444824034038314
V_STDEV=0.6431990734340584

W_MEAN=1.1299844489632986
W_STDEV=0.9414969419163138

MEANS=np.array([R_MEAN,B_MEAN,G_MEAN,N_MEAN])
STDEVS=np.array([R_STDEV,B_STDEV,G_STDEV,N_STDEV])

BRVW_MEANS=np.array([R_MEAN,B_MEAN,V_MEAN,W_MEAN])
BRVW_STDEVS=np.array([R_STDEV,B_STDEV,V_STDEV,W_STDEV])



def norm(img):
    img=(img-MEANS)/STDEVS
    return img[:]


def bgrn_to_brvw_img(img):
    g,r,nir=img[:,:,1], img[:,:,2], img[:,:,3]
    ndwi=(nir-g)/(nir+g)
    del g
    ndvi=(nir-r)/(nir+r)
    del nir
    return np.dstack([img[:,:,0],r,ndvi,ndwi])


def norm_brvw(img):
    img=bgrn_to_brvw_img(img)
    img=(img-BRVW_MEANS)/BRVW_STDEVS
    return img




""" 
#
#   SUBMISSION PROCESSING
#
"""
def load_models(model_names):
    return [load_model(f'models/{name}') for name in model_names]




""" 
#
#   SUBMISSION PROCESSING
#
"""
def generate_submission(
        model,
        filename,
        steps=None,
        data_dir=TEST_DIR,
        lambda_func=norm_brvw,
        batch_size=BATCHSIZE,
        to_tags=True,
        shift=0.0):
    sub_gen=p.DIRGen(
        data_dir,
        lambda_func=lambda_func,
        batch_size=batch_size)
    if to_tags: pp_func=post_process(shift=shift)
    else: pp_func=None
    p.PSubmission.generate_submission(
        filename,
        steps=steps,
        gen=sub_gen,
        model=model,
        to_tags=to_tags,
        pred_processor=pp_func)       

"""
#
# PREDICTION PROCESSING
#
"""
def post_process(shift=0.0,length=17):
    def pp_func(pred):
        return np.round(pred+[shift]*length)
    return pp_func



def test_scoring(gen,model,steps=35,shifts=[0.0,0.1,0.2,0.3,0.35]):
    scorers=[]
    for shift in shifts:
        pgen=p.PredictionGen(
            gen=gen,
            model=model,
            pred_processor=post_process(shift=shift))
        f2scorer=p.FBeta()
        scorers.append(f2scorer)
        print('===>',f2scorer.score_from_gen(pgen,steps))
    return scorers








