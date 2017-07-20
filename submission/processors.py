import os
import yaml
from skimage import io
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from glob import glob
import re
from os.path import basename
from sklearn.metrics import confusion_matrix
import math



"""

    FBeta:  GET FBETA SCORES 

"""
class FBeta(object):
    NOISE_REDUCER=10

    def __init__(self,beta=2,eps=1e-8,noisy=True):
        self.beta=2
        self.eps=eps
        self.noisy=noisy

        
    def _score(self,true,pred):
        tp,fp,_,fn=self._confusion_matrix(true,pred)
        r=self._recall(tp,fp,fn)
        p=self._precision(tp,fp,fn)
        return (1+self.beta**2)*p*r/((self.beta**2)*p + r + self.eps)
        
    
    def _scores(self,trues,preds):
        return [self._score(true,pred) for true,pred in zip(trues,preds)]
        
        
    def score(self,trues,preds):
        trues=np.array(trues)
        preds=np.array(preds)
        if trues.shape!=preds.shape:
            raise ValueError('ERROR: FBeta.score - shapes must match')
        elif len(trues.shape)==1:
            trues=np.expand_dims(trues, axis=0)
            preds=np.expand_dims(preds, axis=0)
        return np.mean(self._scores(trues.tolist(),preds.tolist()),axis=0)


    def score_from_gen(self,gen,steps):
        if self.noisy: print('FBeta.score_from_gen[{}]:'.format(steps))
        self.scores=[]
        for i in range(steps):
            if self.noisy:
                if not (i%self.NOISE_REDUCER): print('\t{}...'.format(i))
            self.scores+=self._scores(*next(gen))
        self.mean_score=np.mean(self.scores,axis=0)
        return self.mean_score


    def _confusion_matrix(self,true,pred):
        m=confusion_matrix(true,pred)     
        return m[1][1],m[0][1],m[0][0],m[1][0]
    

    def _recall(self,tp,fp,fn):
        return tp/(tp + fn + self.eps)
    
    
    def _precision(self,tp,fp,fn):
        return tp/(tp + fp + self.eps)



"""

    PredictionGen:  GET/POST-PROCESS PREDICTIONS

"""
class PredictionGen(object):
    PREDICTION_COLS='truth,prediction\n'
    NOISE_REDUCER=15

    @classmethod
    def generate_submission(
            cls,
            filename,
            gen=None,
            model=None,
            pred_processor=None,
            steps=None,
            noisy=True):
        pgen=cls(
            gen=gen,
            model=model,
            pred_processor=pred_processor)
        steps=steps or math.ceil(gen.size/gen.batch_size)
        if noisy: print('PSubmission.generate_submission[{}*{}]:'.format(steps,gen.batch_size))
        with open(filename,'w') as file:
            file.write(cls.PREDICTION_COLS)
            for i in range(steps):
                if noisy:
                    if not (i%cls.NOISE_REDUCER): print("\t{}...".format(i))
                trues,preds=next(pgen)
                for true,pred in zip(trues,preds):

                    tstr=' '.join(['{:f}'.format(t) for t in true.tolist()])
                    pstr=' '.join(['{:f}'.format(p) for p in pred.tolist()])

                    file.write("{},{}\n".format(tstr,pstr))



    def __init__(self,gen,model,pred_processor=None):
        self.gen=gen
        self.model=model
        self.pred_processor=pred_processor



    def __next__(self):
        imgs,lbls=next(self.gen)
        preds=self.model.predict_on_batch(imgs)
        return lbls,preds 




"""

    THRESHOLDER:

"""

class Thresholder(object):
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


    def __init__(self,csv_path,shifts):
        self.dataframe=pd.read_csv(csv_path)
        self.shifts=shifts
        self.dataframe['prediction']=self._to_list(self.dataframe.prediction)
        self.eps=1e-8


    def tag_dataframe(self,filename=None):
        df=self.dataframe.copy()
        df['tags']=self.preds_to_tags_str(self._thresholded())
        df.drop(['prediction'],axis=1,inplace=True)
        if filename:
            df.to_csv(filename,index=False)
        return df


    def _thresholded(self):
        return self.dataframe.prediction.apply(
            lambda pred: np.round(np.add(pred,self.shifts)).tolist())


    def _to_list(self,dfcolumn):
        return dfcolumn.apply(
            lambda strlist: [float(p) for p in strlist.split(' ')])


    def preds_to_tags_str(self,preds):
        all_tags=[]
        for pred in preds.tolist():
            tags=[]
            for index,value in enumerate(pred):
                if value: tags.append(self.TAGS[index])
            all_tags.append(' '.join(tags))
        return all_tags






class ThresholderFinder(object):


    def __init__(self,csv_path):
        self.dataframe=pd.read_csv(csv_path)
        self.dataframe['truth']=self._to_list(self.dataframe.truth)
        self.dataframe['prediction']=self._to_list(self.dataframe.prediction)
        self.eps=1e-8


    def preds_for_index(self,i):
        return self._for_index(self.dataframe.prediction,i)


    def f2_for_index(self,i,shift=0.0):
        tp,fp,fn=self.tpfpfn_for_index(i,shift)
        p=self._precision(tp,fp,fn)
        r=self._recall(tp,fp,fn)
        return 5*p*r/((4*p)+r+self.eps)


    def score_for_index(self,i,shift=0.0):
        tp,fp,fn=self.tpfpfn_for_index(i,shift)
        return tp-fp-fn


    def best_counts_for_index(self,i,rmin=0.0,rmax=0.5,incr=0.02):
        shifts=np.arange(rmin,rmax,incr)
        shift_f2s=[(shift,self.score_for_index(i,shift)) for shift in shifts]
        bests=[(0,0)]
        for shift_f2 in shift_f2s:
            if shift_f2[1]>bests[0][1]:
                bests=[shift_f2]
            elif shift_f2[1]==bests[0][1]:
                bests.append(shift_f2)
        return bests


    def tpfpfn_for_index(self,i,shift=0.0):
        trues=self.trues_for_index(i)
        # tp,fp        
        preds=self.thresh_preds_for_index(i,shift)
        preds=preds[trues==1]
        nb_trues=preds.shape[0]
        preds=preds[preds==1]
        tp=preds.shape[0]
        fp=nb_trues-tp
        # fn
        preds=self.thresh_preds_for_index(i,shift)
        preds=preds[trues==0]
        preds=preds[preds==1]
        fn=preds.shape[0]
        return tp, fp, fn


    def fn_for_index(self,i,shift=0.0):
        preds=self.thresh_preds_for_index(i,shift)
        trues=self.trues_for_index(i)
        preds=preds[trues==0]
        preds=preds[preds==1]
        fn=preds.shape[0]
        return fn


    def fn_for_index(self,i,shift=0.0):
        preds=self.thresh_preds_for_index(i,shift)
        preds=preds[preds==self.trues_for_index(i)]
        return preds.shape[0]

    def true_count_for_index(self,i,shift=0.0):
        preds=self.thresh_preds_for_index(i,shift)
        preds=preds[preds==self.trues_for_index(i)]
        return preds.shape[0]


    def thresh_preds_for_index(self,i,shift=0.0):
        preds=self.preds_for_index(i)
        return preds.add(shift).round()


    def trues_for_index(self,i):
        return self._for_index(self.dataframe.truth,i)


    def _for_index(self,dfcolumn,i):
        return dfcolumn.apply(
            lambda value_list: value_list[i])


    def _to_list(self,dfcolumn):
        return dfcolumn.apply(
            lambda strlist: [float(p) for p in strlist.split(' ')])


    def _recall(self,tp,fp,fn):
        return tp/(tp + fn + self.eps)
    
    
    def _precision(self,tp,fp,fn):
        return tp/(tp + fp + self.eps)


"""

    PSubmission:  GENERATE SUBMISSIONS

"""
class PSubmission(object):
    NOISE_REDUCER=15
    SUBMISSION_COLS='image_name,tags\n'
    PREDICTION_COLS='image_name,predictions\n'
    """
    image_name,tags
    test_0,agriculture road water
    test_1,primary clear
    test_2,haze primary
    """
    
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

    
    @classmethod
    def generate_submission(
            cls,
            filename,
            gen=None,
            model=None,
            pred_processor=None,
            to_tags=True,
            steps=None,
            noisy=True):
        psub=cls(
            gen=gen,
            model=model,
            pred_processor=pred_processor,
            to_tags=to_tags)
        steps=steps or math.ceil(gen.size/gen.batch_size)
        if noisy: print('PSubmission.generate_submission[{}*{}]:'.format(steps,gen.batch_size))
        with open(filename,'w') as file:
            file.write(cls.SUBMISSION_COLS)
            for i in range(steps):
                if noisy:
                    if not (i%cls.NOISE_REDUCER): print("\t{}...".format(i))
                names,tags=next(psub)
                for name,tag in zip(names,tags):
                    file.write("{},{}\n".format(name,tag))

                    
    def __init__(self,gen,model,pred_processor=None,to_tags=True):
        self.gen=gen
        self.model=model
        self.pred_processor=pred_processor
        self.to_tags=to_tags

        
    def preds_to_tags_str(self,preds):
        all_tags=[]
        for pred in preds:
            tags=[]
            for index,value in enumerate(pred):
                if value: tags.append(self.TAGS[index])
            all_tags.append(' '.join(tags))
        return all_tags
                
        
    def __next__(self):
        imgs,names=next(self.gen)
        if imgs.any():
            preds=self.model.predict_on_batch(imgs)
            if self.pred_processor: 
                preds=self.pred_processor(preds)
            if self.to_tags:
                return names,self.preds_to_tags_str(preds)
            else:
                str_preds=[]
                for pred in preds:
                    str_preds.append(
                        ' '.join(['{:f}'.format(p) for p in pred.tolist()]))
                return names,str_preds










"""

    DIRGen:  FLOW FROM DIR WITH LAMBDA FUNC 

"""
class DIRGen(object):
    
    def __init__(self,directory,lambda_func=None,file_ext='tif',strip_ext=True,batch_size=16):
        self.directory=directory
        self.file_ext=file_ext
        self.strip_ext=strip_ext
        self.paths=sorted(glob('{}/*.{}'.format(directory,file_ext)))
        self.batch_size=batch_size
        self.lambda_func=lambda_func
        if self.strip_ext:
            self.names=list(map(
                lambda path: re.sub('.{}$'.format(self.file_ext),'',basename(path)),
                self.paths))
        else:
            self.names=list(map(lambda path: basename(path),self.paths))
        self.batch_index=0
        self.stop=False
        self.size=len(self.paths)
            
            
    
    def __next__(self):
        if self.stop:
            print('STOPPPPPPP!')
            return np.array([]),np.array([])
        else:
            start=self.batch_index*self.batch_size
            end=start+self.batch_size
            if (end>=self.size): 
                end=self.size
                self.stop=True
            batch_names=self.names[start:end]
            batch_paths=self.paths[start:end]
            batch_imgs=[self._img_data(path) for path in batch_paths]
            self.batch_index+=1
            return np.array(batch_imgs),np.array(batch_names)

    
    
    def _img_data(self,path):
        """Read Data 
        """
        im=io.imread(path)
        if self.lambda_func:
            return self.lambda_func(im)
        else:
            return im




"""

    MultiModel: Predict from multiple models
    # MAP {Target: CURRENT}

    TARGET INDICES:
        0 primary
        1 clear
        2 agriculture
        3 road
        4 water
        5 partly_cloudy
        6 cultivation
        7 habitation
        8 haze
        9 cloudy
        10 bare_ground
        11 selective_logging
        12 artisinal_mine
        13 blooming
        14 slash_burn
        15 conventional_mine
        16 blow_down

    MODEL INPUTS:
        0: 0 primary
        1: 2 agriculture
        2: 3 road
        3: 4 water
        4: 6 cultivation
        5: 7 habitation
        weather: 
            6: 1 clear
            7: 5 partly_cloudy
            8: 8 haze
            9: 9 cloudy
        rare:
            10: 10 bare_ground
            11: 11 selective_logging
            12: 12 artisinal_mine
            13: 13 blooming
            14: 14 slash_burn
            15: 15 conventional_mine
            16: 16 blow_down


    MAP:
        model_map={
            0: 0,
            1: 2,
            2: 3,
            3: 4,
            4: 6,
            5: 7,
            6: 1,
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
"""

class MultiModel(object):
    def __init__(self,models,model_map=None):
        self.models=models
        if model_map:
            self.model_map=model_map
            self.output_size=len(model_map)
    
    def predict_on_batch(self,batch):
        models_preds=[]
        for model in self.models:
            models_preds.append(model.predict_on_batch(batch))
        preds=np.concatenate(models_preds,axis=1)
        if self.model_map:
            preds=self._map_preds(preds)
        return preds
    
    def _map_preds(self,preds):
        mapped_preds=[]
        for pred in preds:
            mapped_preds.append([pred[self.model_map[i]] for i in range(self.output_size)])
        return np.array(mapped_preds)





"""

    TagTrans:  LABELS TO TAGS

"""
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


    
class TagTrans(object):
    def __init__(self,gen):
        self.gen=gen
        
    def __next__(self):
        trues,preds=next(self.gen)
        return trues,self._preds_to_tags_str(preds)
    
    def _preds_to_tags_str(self,preds):
        all_tags=[]
        for pred in preds:
            tags=[]
            for index,value in enumerate(pred):
                if value: tags.append(TAGS[index])
            all_tags.append(' '.join(tags))
        return all_tags