import sys
from .constants import *
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from ..exception import CustomException
from ..logger import logging

class GetData():
    def __init__(self, data_dir:str=TRAIN_DATA_DIR,
                 subset:str='training', 
                 validation_split:float=0.2) -> None:
        """
        Returns tensorflow image_dataset_from_directory instance.
        
        Args:
            - data_dir: path of the dataset folder
            - subset: "training" or "validation" or None
            - validation_split: (0, 1.0) or None
        """
        self.data_dir = data_dir
        self.subset = subset
        self.validation_split = validation_split

    def get_data(self):
        try:
            data_ds = image_dataset_from_directory(directory=self.data_dir,
                                                    label_mode='categorical',
                                                    batch_size=BATCH_SIZE,
                                                    image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    validation_split=self.validation_split,
                                                    seed=SEED,
                                                    subset=self.subset)
            
            return data_ds
        except Exception as e:
            raise CustomException(e, sys) from e


    def data(self):
        try:
            AUTOTUNE = tf.data.AUTOTUNE

            data_ds = self.get_data()
            class_names = data_ds.class_names
            data_ds = data_ds.cache().shuffle(buffer_size=1000).prefetch(buffer_size=AUTOTUNE)

            if self.subset is not None:
                logging.info(f"{self.subset.capitalize()} Image Dataset loaded and cached using tensorflow image_dataset_from_directory")
            else:
                logging.info(f"Test Image Dataset loaded and cached using tensorflow image_dataset_from_directory")

            return data_ds, class_names
        except Exception as e:
            raise CustomException(e, sys) from e
