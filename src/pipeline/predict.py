import os, sys
import random
import numpy as np
import tensorflow as tf
from .constants import *
from . import utils
from .get_data import GetData
from ..logger import logging
from ..exception import CustomException



class Predict():
    def __init__(self) -> None:
        try:
            model_name = os.listdir(MODEL_DIR)[-2] # latest trained model name
            self.model_path = os.path.join(MODEL_DIR, model_name)
            self.test_data_dir = TEST_DATA_DIR
        except Exception as e:
            raise CustomException(e, sys) from e

    def test_data(self):
        try:
            get_test_data = GetData(data_dir=self.test_data_dir, 
                                    subset=None, validation_split=None)
            
            test_ds = get_test_data.get_data()

            logging.info(f"Test Image Dataset loaded using tensorflow image_dataset_from_directory")

            return test_ds
        except Exception as e:
            raise CustomException(e, sys) from e

    def batch_predict(self) -> None:
        """
        Predict on Test Data
        """
        try:
            model = utils.load_model(self.model_path)
            results = model.evaluate(self.test_data())
            print(f"Loss: {results[0]: .4f}, Accuracy: {results[1]*100 :.2f}%")

            logging.info("Batch Prediction done on test dataset. Below are the Loss & Accuracy")
            logging.info(f"Loss: {results[0]: .4f}, Accuracy: {results[1]*100 :.2f}%")
            return results
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def predict_random_test_images(self, image_path=None) -> dict:
        """
        if image_path is None, it randomly picks an image from test data directory.

        Return a dictionary with actual and predicted class along with confidence.
        """

        # Load test data and saved model
        try:
            test_ds = self.test_data()
            class_names = test_ds.class_names
            model = utils.load_model(self.model_path)

            # get random image
            if image_path is None:
                image_path = random.sample(test_ds.file_paths, 1)[0]

            actual_img_class = os.path.basename(os.path.dirname(image_path))

            img = tf.keras.utils.load_img(image_path, 
                                        target_size=(IMG_HEIGHT, IMG_WIDTH))

            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            output = {"Actual Image Class": actual_img_class, 
                    "Predicted Image Class": class_names[np.argmax(score)],
                    "Confidence": 100 * np.max(score)}
            
            print("Actual Image Class: [{}]\nPredicted Image Class: [{}]\nConfidence {:.2f}%"
                .format(actual_img_class, class_names[np.argmax(score)], 100 * np.max(score))
                )
            
            logging.info(f"Prediction on random test images is completed.\nImage Path: [{image_path}]")
            logging.info(f"Result: [{output}]")

            return output
        except Exception as e:
            raise CustomException(e, sys) from e
