from skimage import io
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
class CSVGen():
    NAME_COLUMN='image_name'
    PATH_COLUMN='image_path'
    LABEL_COLUMN='vec'
    LABEL_TO_LIST=True

    def __init__(self,file,image_ext='tif',image_dir=None,batch_size=32):
        """Initialize Generator
            Args:
                file: <str> path to 2-column csv 
                file_type: <str> file ext 
                batch_size: <int> batch-size
        """
        self.file=file
        self.batch_size=batch_size
        self.image_ext=image_ext
        self.image_dir=image_dir
        self._set_data()



    def data(self):
        """Data Generator
            batchwise return tuple of (images,labels)
        """        
        batch_index=0
        while True:
            start=batch_index*self.batch_size
            if ((start+self.batch_size)>=self.size):
                self.labels, self.paths = shuffle(self.labels,self.paths)
                batch_index=0
            batch_labels=self.labels[start:start+self.batch_size]
            batch_paths=self.paths[start:start+self.batch_size]
            batch_imgs=[self._imdata(img) for img in batch_paths]
            yield np.array(batch_imgs),np.array(batch_labels)
            batch_index+=1
    
    
    #
    # INTERNAL METHODS
    #
    def _imdata(self,path):
        """Read Data
            Args:
                path: <str> path to image
        """
        return io.imread(path)
    
    
    def _set_data(self):
        """Set Data
            sets two instance properties:
                self.labels
                self.paths
            the paths and labels are pairwised shuffled
        """
        df=pd.read_csv(self.file,sep=' ')
        self.size=df.shape[0]
        df[self.PATH_COLUMN]=df[self.NAME_COLUMN].apply(self._image_path_from_name)
        if self.LABEL_TO_LIST: 
            df[self.LABEL_COLUMN]=df[self.LABEL_COLUMN].apply(self._strlist_to_list)
        labels=df[self.LABEL_COLUMN].values.tolist()
        paths=df[self.PATH_COLUMN].values.tolist()
        self.labels, self.paths = shuffle(labels,paths)


    def _image_path_from_name(self,name):
        """ Get image path from image name
            Args:
                name: <str> name of image file (without ext)
        """
        return f'{self.image_dir}/{name}.{self.image_ext}'


    def _strlist_to_list(self,strlist):
        """ Convert a list in string form to a list
        """
        return list(eval(strlist))

