import os
import subprocess
from src.pipeline.get_data import GetData
from src.pipeline.train import BuildModel
from src.pipeline.constants import *
from src.pipeline.utils import save_model
from src.pipeline.predict import Predict


val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

if val == 0:
    # Get training and validation data
    train_ds, class_names = GetData().data()
    val_ds, _ = GetData(subset='validation').data()

    num_classes = len(class_names)

    build_model = BuildModel(num_classes=num_classes)
    model = build_model.build_cnn_model()

    history = build_model.fit(model, 
                              train_ds, val_ds, 
                              epochs=EPOCHS)
    
    model_path = save_model(model)
    print(f"Model saved in: [{model_path}]")
elif val == 1:
    print("Predicting on All Test Data")
    result = Predict().batch_predict()

    print("Predicting on random test-image")
    output = Predict().predict_random_test_images()
else:
    # For prod deployment
    '''process = subprocess.Popen(['sh', './src/pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )'''

    # For dev deployment
    process = subprocess.Popen(['python', './src/pipeline/deploy.py'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True
                                )

    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)