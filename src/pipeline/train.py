import sys
import tensorflow as tf
from keras import layers, models
from src.pipeline.constants import INPUT_SHAPE, EPOCHS
from src.exception import CustomException
from src.logger import logging

tf.get_logger().setLevel('ERROR')


class BuildModel():
    def __init__(self, num_classes=3, input_shape=INPUT_SHAPE):
        super(BuildModel, self).__init__()

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.data_augmentation = models.Sequential()
        self.model = models.Sequential()

    def build_data_augmentation(self):
        self.data_augmentation.add(layers.InputLayer(input_shape=self.input_shape))
        self.data_augmentation.add(layers.RandomFlip(mode='horizontal'))
        self.data_augmentation.add(layers.RandomRotation(0.1))
        self.data_augmentation.add(layers.RandomZoom(0.1))
        self.data_augmentation.add(layers.Rescaling(1./255))
        
        return self.data_augmentation

    def build_cnn_model(self):
        """
        Returns compiled model object.
        """
        self.model.add(self.build_data_augmentation())

        self.model.add(layers.Convolution2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D())

        self.model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D())

        self.model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D())
        # self.model.add(layers.Dropout(0.2))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(self.num_classes, activation='softmax')) # activation='softmax'


        # Comilpile the model
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy', 
                           metrics=['accuracy'])
        
        return self.model
    
    def fit(self, model, train_ds, val_ds, epochs=EPOCHS):
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=epochs,
                            workers=10)
        return history


 

