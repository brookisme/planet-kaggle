import pickle
from keras.callbacks import Callback

###################################################################
#
# TrainHistory <keras callback>: 
#
#    Capture loss, accuracy, f2 score during training
#
####################################################################

class LossAccF2Measures(object):
    def __init__(self):
        self.loss = []
        self.f2_score = []
        self.accuracy = []

    def dict(self):
        return {
            'loss': self.loss,
            'f2_score': self.f2_score,
            'accuracy': self.accuracy}




####################################################################
#
# TrainHistory <keras callback>: 
#
#    Capture loss, accuracy, f2 score during training
#
####################################################################

class LossAccF2History(Callback):

    def __init__(self,save_path=None,**kwargs):
        """ 
            Args:
                save_path:  <str> file path to save history to 
                            pickle at the end of each epoch
        """
        self.save_path=save_path
        super().__init__(**kwargs)


    #
    #  CALLBACK METHODS
    #
    def on_train_begin(self, logs={}):
        """ init measures for training/validation
        """
        self.train=LossAccF2Measures()
        self.valid=LossAccF2Measures()


    def on_batch_end(self, batch, logs={}):
        """ capture training measures
        """
        self.train.loss.append(logs.get('loss'))
        self.train.f2_score.append(logs.get('k_f2'))
        self.train.accuracy.append(logs.get('acc'))
 

    def on_epoch_end(self, epoch, logs={}):
        """ capture validation measures
            - save to self.save_path if exists
        """
        self.valid.loss.append(logs.get('val_loss'))
        self.valid.f2_score.append(logs.get('val_k_f2'))
        self.valid.accuracy.append(logs.get('val_acc'))        
        if self.save_path: self._save()


    #
    #  INTERNAL
    #
    def _save(self):
        """ save train/valid dictionaries
        """
        self._save_obj(f'{self.save_path}.train.p',self.train.dict())
        self._save_obj(f'{self.save_path}.valid.p',self.valid.dict())


    def _save_obj(self,path,obj):
        """ save object to pickle
        """
        with open(path, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

