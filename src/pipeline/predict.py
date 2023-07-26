import os
import random
import numpy as np
import tensorflow as tf
from src.pipeline.constants import *
from src.pipeline import utils
from src.pipeline.get_data import GetData


class Predict():
    def __init__(self) -> None:
        model_name = os.listdir(MODEL_DIR)[-2] # latest trained model name
        self.model_path = os.path.join(MODEL_DIR, model_name)
        self.test_data_dir = TEST_DATA_DIR

    def test_data(self):
        get_test_data = GetData(data_dir=self.test_data_dir, 
                                subset=None, validation_split=None)
        
        test_ds = get_test_data.get_data()
        return test_ds

    def batch_predict(self) -> None:
        """
        Predict on Test Data
        """
        model = utils.load_model(self.model_path)
        results = model.evaluate(self.test_data())
        print(f"Loss: {results[0]: .4f}, Accuracy: {results[1]*100 :.2f}%")
        return results
    
    def predict_random_test_images(self, image_path=None) -> dict:
        """
        if image_path is None, it randomly picks an image from test data directory.

        Return a dictionary with actual and predicted class along with confidence.
        """

        # Load test data and saved model
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
        
        return output
