import os
from importlib import reload
import processors as p
import helpers as h
import dfgen as dfg
from keras.models import load_model

reload(p)
reload(h)
reload(dfg)


""" GET FILES

gcloud config set project gpu-services
gcloud compute copy-files gpu-82-2:~/kaggle/planet/planet-kaggle/models/graph/res_bn_brvw/models/* models/
gcloud config set project bgw-gpu
gcloud compute copy-files gpu-82:~/kaggle/planet/planet-kaggle/models/graph/res_bn_brvw/models/* models/
gcloud compute copy-files gpu-82-a:~/kaggle/planet/planet-kaggle/models/graph/res_bn_brvw/models/* models/
cp ~/kaggle/planet/planet-kaggle/models/graph/res_bn_brvw/models/* models/



gcloud compute copy-files gpu-82-a:~/kaggle/planet/planet-kaggle/submission/* .

# GET FILES
gcloud compute copy-files gpu-82-a:/data/planet/test-tif/* test-tif/



gcloud compute copy-files gpu-82-a:~/kaggle/planet/planet-kaggle/submission/res17_model-{shift}.csv .


gcloud compute copy-files gpu-82-a:~/kaggle/planet/planet-kaggle/submission/preds/res17_thresholder_1.csv .

"""


#
# SETUP
#
BATCHSIZE=64
REPO_DIR=f'{os.environ["PKR"]}'
DATA_DIR=f'{os.environ["DATA"]}/planet'
TEST_DIR=f'{DATA_DIR}/test-tif'
TRAIN_DIR=f'{DATA_DIR}/train-tif'
TRAIN_CSV=f'{REPO_DIR}/datacsvs/train.csv'
VALID_CSV=f'{REPO_DIR}/datacsvs/valid.csv'


train_gen=dfg.DFGen(
    csv_file=TRAIN_CSV,
    csv_sep=',',
    batch_size=BATCHSIZE,
    lambda_func=h.norm_brvw)


valid_gen=dfg.DFGen(
    csv_file=VALID_CSV,
    csv_sep=',',
    batch_size=BATCHSIZE,
    lambda_func=h.norm_brvw)




"""
#
# MODELS
#
"""


#
#   17vector
#

model_name='bn_res_brvw_17.hdf5'
res17_model=load_model(f'models/{model_name}')



res17_model.summary()



reload(p)
reload(h)
reload(dfg)

p.PredictionGen.generate_submission('preds/res17_valid-pred.csv',valid_gen,res17_model)



sub_gen=p.DIRGen(
    TEST_DIR,
    batch_size=2)
psub=p.PSubmission(valid_gen,res17_model,to_tags=False)
test_t,test_p=next(psub)
test_p[0]


res17_scorers=h.test_scoring(valid_gen,res17_model,shifts=[0.35,0.325])


shift=0.35
h.generate_submission(
    res17_model,
    f'res17_model-{shift}.csv',
    shift=shift)



h.generate_submission(
    res17_model,
    f'preds/res17_model-pred.csv',
    to_tags=False)

#
# THRESHAER
#
tf=p.ThresholderFinder('preds/res17_valid-pred.csv')
shifts=[]
for i in range(17):
    print("\n\n\nINDEX {}:\n".format(i))
    bests=t.best_counts_for_index(i)
    print("\t{}".format(bests))
    shifts.append(bests[0][0])


#
# VALID RESULT
#

from importlib import reload
import processors as p
reload(p)

shifts=[0.0, 0.0, 0.06, 0.0, 0.02, 0.06, 0.10, 0.04, 0.02, 0.0, 0.22, 0.32, 0.14, 0.42, 0.40, 0.42, 0.48]
t=p.Thresholder('preds/res17_model-pred.csv',shifts)


tag_df=t.tag_dataframe('preds/res17_thresholder_1.csv')



#
# OTHERS
#
model_names=[
    'primary_1024x512_0.hdf5',
    'agriculture_1024x512_1.hdf5',
    'road_1024x512_1.hdf5',
    'water_1024x512_1.hdf5',
    'cultivation_1024x512_0.hdf5',
    'habitation_1024x512_2.hdf5',
    'weather_1024x512_3.hdf5',
    'rare_1024x512_1.hdf5', 
]

for model_name in model_names[:5]:
    name=model_name.replace('hdf5','pred.csv')
    print(f'\n\n\nMODEL: {name}\n')
    model=load_model(f'models/{model_name}')
    h.generate_submission(
        model,
        f'preds/{name}',
        to_tags=False)


#
# MULTI-MODEL
#

# TODO REPLACE WITH REAL HDF5 NAMES
model_names=[
    'primary_1024x512_0.hdf5',
    'agriculture_1024x512_1.hdf5',
    'road_1024x512_1.hdf5',
    'water_1024x512_1.hdf5',
    'cultivation_1024x512_0.hdf5',
    'habitation_1024x512_2.hdf5',
    'weather_1024x512_3.hdf5',
    'rare_1024x512_1.hdf5', 
]


model_map={
    0: 0,
    1: 6,
    2: 1,
    3: 2,
    4: 3,
    5: 7,
    6: 4,
    7: 5,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16
}

models=h.load_models(model_names)
multi_model=p.MultiModel(models,model_map)


# multi_scorers=h.test_scoring(valid_gen,multi_model,steps=10)


# i,l=next(valid_gen)
# p=multi_model.predict_on_batch(i)
# l
# np.round(p)








"""
#
#   COMBO-MODEL
#
"""
# TODO REPLACE WITH REAL HDF5 NAMES
model_name='combo'
combo_model=load_model(f'models/{model_name}')
combo_scorers=h.test_scoring(valid_gen,combo_model)

h.generate_submission(
    combo_model,
    'combo_model-1.csv',
    shift=0.0)







