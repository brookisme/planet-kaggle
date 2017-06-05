import os
from skimage import io
from sklearn.utils import shuffle
import pandas as pd
import numpy as np


DATA_ROOT=os.environ.get('DATA')
DATA_DIR='planet'
ROOT=f'{DATA_ROOT}/{DATA_DIR}'
JPG_DIR = os.path.join(ROOT, 'train-jpg')
TIF_DIR = os.path.join(ROOT, 'train-tif')

class DFGen():
    """ CREATES GENERATOR FROM DATAFRAME

        Usage:
            train_gen=DFGen(
                dataframe=train_dataframe,batch_size=128)
            next(train_gen)

        Args:
            dataframe: <dataframe> dataframe with label and image_path column
            file: <str> (if not dataframe) path to csv with labels/image_paths
            image_ext: <str> img ext (tif|jpg)
            image_dir: <str> directory where images exist
            batch_size: <int> batch-size
    
    """
    NAME_COLUMN='image_name'
    PATH_COLUMN='image_path'
    LABEL_COLUMN='vec'

    def __init__(
            self,
            file=None,
            dataframe=None,
            image_ext='tif',
            image_dir=None,
            ndvi_images=False,
            batch_size=32):
        self.batch_index=0
        self.file=file
        self.batch_size=batch_size
        self.image_ext=image_ext
        self.ndvi_images=ndvi_images
        self._set_image_dir(image_dir)
        self._set_data(file,dataframe)


    def __next__(self):
        """
            batchwise return tuple of (images,labels)
        """        
        start=self.batch_index*self.batch_size
        end=start+self.batch_size
        if (end>=self.size):
            self.labels, self.paths = shuffle(self.labels,self.paths)
            self.batch_index=0
        batch_labels=self.labels[start:end]
        batch_paths=self.paths[start:end]
        batch_imgs=[self._imdata(img) for img in batch_paths]
        self.batch_index+=1
        return np.array(batch_imgs),np.array(batch_labels)
    
    
    #
    # INTERNAL METHODS
    #
    def _imdata(self,path):
        """Read Data
            Args:
                path: <str> path to image
        """
        img=io.imread(path)
        if self.ndvi_images:
            return self._ndviimg(img)
        else:
            return img
    

    def _ndvi(self,img):
        r=img[:,:,0]
        nir=img[:,:,3]
        return (nir-r)/(nir+r)



    def _ndviimg(self,img):
        ndvi_band=self._ndvi(img)
        img[:,:,3]=ndvi_band
        return img


    def _set_image_dir(self,image_dir):
        """Set Image Dir
        """
        if image_dir: self.image_dir=image_dir
        else:
            if self.image_ext=='tif':
                self.image_dir=TIF_DIR
            else:
                self.image_dir=JPG_DIR


    def _set_data(self,file,df):
        """Set Data
            sets three instance properties:
                self.labels
                self.paths
                self.dataframe
            the paths and labels are pairwised shuffled
        """
        if (df is None) or (df is False): 
            df=pd.read_csv(self.file,sep=' ')
        self.size=df.shape[0]
        df[self.PATH_COLUMN]=df[self.NAME_COLUMN].apply(self._image_path_from_name)
        labels=df[self.LABEL_COLUMN].values.tolist()
        paths=df[self.PATH_COLUMN].values.tolist()
        self.labels, self.paths = shuffle(labels,paths)
        self.dataframe=df


    def _image_path_from_name(self,name):
        """ Get image path from image name
            Args:
                name: <str> name of image file (without ext)
        """
        return f'{self.image_dir}/{name}.{self.image_ext}'


