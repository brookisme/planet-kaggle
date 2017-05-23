import os
import math
import pandas as pd



DATA_ROOT=os.environ.get('DATA')
DATA_DIR='planet-old'
ROOT=f'{DATA_ROOT}/{DATA_DIR}'
LABEL_CSV=os.path.join(ROOT,'train.csv')

class PlanetData(object):
    """ CONSTRUCTS TRAINING/VALIDATION DATAFRAMES TO BE USED IN CSVGenerator

        Args:
            - train_size:
                <int> number of training examples
                <str> 'full' based on full use of dataset
            - labels_df: Labels dataframe
            - csv_path: (if labels_df not given) path to labels CSV
            - valid_size: <int> number of validation examples
            - valid_pct: 
                <int> (if valid_size not given) valid_size=train_size * valid_pct / 100
            - version: <str|int> string used naming of csv_files
            - create:
                - if True:
                    create the train/valid dataframes/CSVs
                    if auto_save: automatically save the train/valid CSVs
                    if auto_clear: set labels_df to None after creation of train/valid
                - else:
                    load the train/valid dataframes for the parameters above
    
    """
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
            labels_df=None,
            version=1,
            create=False,
            auto_save=True,
            auto_clear=True):
        self._init_params()
        self.version=version
        self.auto_save=auto_save
        self.auto_clear=auto_clear
        if create:
            self._set_label_df(labels_df,csv_path)
            self._set_df_sizes(train_size,valid_size,valid_pct)
            self._set_train_test_dfs()
        else:
            self._set_df_sizes(train_size,valid_size,valid_pct)
            self.load_dataframes()


    def train_path(self):
        """ Path to trainng CSV
        """
        if self.is_full:
            name=f'training_data_v{self.version}.csv'
        else:
            name=f'training_data_{self.train_size}_v{self.version}.csv'
        return f'{ROOT}/{name}'

    
    def valid_path(self):
        """ Path to validation CSV
        """
        if self.is_full:
            name=f'validation_data_v{self.version}.csv'
        else:
            name=f'validation_data_{self.valid_size}_v{self.version}.csv'
        return f'{ROOT}/{name}'


    def load_dataframes(self):
        """ Load train/valid CSVs
        """        
        self.train_df=pd.read_csv(self.train_path(),sep=' ')
        self.valid_df=pd.read_csv(self.valid_path(),sep=' ')


    def save(self):
        """ Save train/valid CSVs
        """              
        self.train_df.to_csv(self.train_path(),index=False,sep=' ')
        self.valid_df.to_csv(self.valid_path(),index=False,sep=' ')


    #
    # INTERNAL
    #
    def _init_params(self):
        self.train_size=None
        self.labels_df=None
        self.is_full=True


    def _set_label_df(self,labels_df,csv_path):
        self.labels_df=labels_df or pd.read_csv(csv_path)
        self.labels_df['vec']=self.labels_df.tags.apply(self._tags_to_vec)


    def _set_train_test_dfs(self):
        self.train_df=self.labels_df.sample(self.train_size)
        self.valid_df=self.labels_df.drop(
            self.train_df.index).sample(self.valid_size)
        if self.auto_save: self.save()
        if self.auto_clear: self.labels_df=None


    def _set_df_sizes(self,train_size,valid_size,valid_pct):
        """ set sizes for training and validation set
            -   if train size is None, or a string ~ 'FULL' 
                the whole dataset (minus the validation set) is used
            -   valid_size is used over valid_pct
        """
        if type(train_size) is int:
            self.train_size=train_size
            self.is_full=False
        elif self.labels_df is not None:
            nb_rows=self.labels_df.shape[0]
            if not valid_size:
                valid_size=math.floor(nb_rows*valid_pct/100)
            self.train_size=nb_rows-valid_size
        if valid_size: 
            self.valid_size=valid_size
        if self.train_size: 
            self.valid_size=math.floor(self.train_size*valid_pct/100)
    


    def _tags_to_vec(self,tags):
        """ Convert Tags to a List Vector
            - list ordering given by TAGS property
        """     
        tags=tags.split(' ')
        return [int(label in tags) for label in self.TAGS]





