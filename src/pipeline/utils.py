import os
import tensorflow as tf
from datetime import datetime
from src.pipeline.constants import MODEL_DIR


def save_model(model: tf.keras.models.Sequential):
    CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    model_name = CURRENT_TIME_STAMP + "_CNN-Model.h5"
    model_path = os.path.join(MODEL_DIR, model_name)

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(model_path)
    return model_path


def load_model(model_path):
    model = None
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        print("Please enter correct path")
        # print(model_path)
        exit(0)

    return model