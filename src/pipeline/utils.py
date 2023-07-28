import os, sys
import tensorflow as tf
from datetime import datetime
from .constants import MODEL_DIR
from ..exception import CustomException
from ..logger import logging



def save_model(model: tf.keras.models.Sequential):
    try:
        CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        model_name = CURRENT_TIME_STAMP + "_CNN-Model.h5"
        model_path = os.path.join(MODEL_DIR, model_name)

        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save(model_path)

        logging.info(f"Model Saved at [{model_path}]")
        return model_path
    except Exception as e:
        raise CustomException(e, sys) from e


def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        
        logging.info(f"Model Loaded Successfully from [{model_path}]")
        return model
    except:
        print("Please enter correct path")
        exit(0)