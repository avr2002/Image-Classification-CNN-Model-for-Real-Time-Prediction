from flask import Flask, request
from src.pipeline.predict import Predict
# from src.pipeline.utils import load_model
# from src.pipeline.constants import *


app = Flask(__name__)


input_image_path = "./output/api_input.jpg"

@app.post("/get-image-class")
def get_image_class():
    image = request.files['file']
    image.save(input_image_path)
    output = Predict().predict_random_test_images(image_path=input_image_path)

    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)