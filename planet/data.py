import os
import math
import pandas as pd



DATA_ROOT=os.environ.get('DATA')
DATA_DIR='planet-old'
ROOT=f'{DATA_ROOT}/{DATA_DIR}'
LABEL_CSV=os.path.join(ROOT,'train.csv')

class PlanetData(object):

    TAGS=[
        'artisinal_mine',
        'haze',
        'water',
        'conventional_mine',
        'agriculture',
        'cultivation',
        'bare_ground',
        'primary',
        'habitation',
        'partly_cloudy',
        'blooming',
        'blow_down',
        'slash_burn',
        'road',
        'clear',
        'selective_logging',
        'cloudy']


    def __init__(self,
            train_size=200,
            valid_pct=20,
            valid_size=None,
            csv_path=LABEL_CSV,
            version=1,
            read=False,
            auto_save=True):
        # set sizes
        self.train_size=train_size
        if valid_size: self.valid_size=valid_size
        else: self.valid_size=math.floor(train_size*valid_pct/100)
        # other params
        self.version=version
        self.auto_save=auto_save
        # dataframes
        if read:
            self.load_dataframes()
        else:
            self._set_dataframes(csv_path)


    def train_path(self):
        name=f'training_data_{self.train_size}_v{self.version}.csv'
        return f'{ROOT}/{name}'

    
    def valid_path(self):
        name=f'validation_data_{self.valid_size}_v{self.version}.csv'
        return f'{ROOT}/{name}'


    def load_dataframes(self):
        self.train_df=pd.read_csv(self.train_path(),sep=' ')
        self.valid_df=pd.read_csv(self.valid_path(),sep=' ')


    def save(self):
        self.train_df.to_csv(self.train_path(),index=False,sep=' ')
        self.valid_df.to_csv(self.valid_path(),index=False,sep=' ')


    #
    # INTERNAL
    #
    def _set_dataframes(self,csv_path):
        self.labels_df=pd.read_csv(csv_path)
        self.labels_df['vec']=self.labels_df.tags.apply(self._tags_to_vec)
        self.train_df=self.labels_df.sample(self.train_size)
        self.valid_df=self.labels_df.drop(
            self.train_df.index).sample(self.valid_size)
        if self.auto_save:
            self.save()


    def _tags_to_vec(self,tags):
        tags=tags.split(' ')
        return [int(label in tags) for label in self.TAGS]





