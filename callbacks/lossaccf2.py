import pickle
from callbacks.measures import HistoryMeasures
from keras.callbacks import Callback




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
        self.batch=HistoryMeasures(
            ['loss','k_f2','acc'])
        self.epoch=HistoryMeasures(
            ['loss','k_f2','acc','val_loss','val_k_f2','val_acc'])


    def on_batch_end(self, batch, logs={}):
        """ capture training measures
        """
        self.batch.update(logs)
 

    def on_epoch_end(self, epoch, logs={}):
        """ capture validation measures
            - save to self.save_path if exists
        """
        self.epoch.update(logs)      
        if self.save_path: self._save()


    #
    #  INTERNAL
    #
    def _save(self):
        """ save train/valid dictionaries
        """
        self._save_obj(f'{self.save_path}.batch.p',self.batch.dict())
        self._save_obj(f'{self.save_path}.epoch.p',self.epoch.dict())


    def _save_obj(self,path,obj):
        """ save object to pickle
        """
        with open(path, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

