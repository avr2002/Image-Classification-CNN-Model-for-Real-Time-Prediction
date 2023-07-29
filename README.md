# Image-Classification-CNN-Model-for-Real-Time-Prediction

## Objective
To implement Convolutional Neural Network for Multi-Class Classification to classify images into `driving license`, `social security`, and `others` category.


## Tech Stack
- $\rightarrow$ **Language:** Python
- $\rightarrow$ **Libraries:** `tensorflow`, `keras`, `matplotlib`, `flask`, `gunicorn`, `pathlib`, `numpy`

## Output
$\rightarrow$ An accuracy of 96% was achieved on test data of 150 images.

<div align='center'>
    <img src = "output/prediction_3.png"/>
    <img src = "output/prediction_4.png"/>
</div>


## Dataset Summary
Dataset used in this project are images of driving license, social security, and others categorized into respective categories. The images are of different shapes and sizes which are preprocessed before modeling.

## Approach
1. Data loading
2. Data Preprocessing
3. Data Augmentation
4. Model building and training
5. Deployment using `flask`, `gunicorn`

## Folder Structure

```
input
    |__test_data
    |__train_data
        |__driving_license
        |__others
        |__social_security
logs
    |__log_files
notebooks
    |__images
    |__CNN.ipynb (**MAIN NOTEBOOK**)
    |__helper_functions.py
output
    |__saved_models
    |__prediction_images
src
    |__exception
        |__`__init__.py`
    |__logger
        |__`__init__.py`
    |__pipeline
        |__constants.py
        |__deploy.py
        |__get_data.py
        |__predict.py
        |__train.py
        |__utils.py
        |__wsgi.py
        |__wsgi.sh
engine.py
model_api.ipynb
```

## Project Takeaways

1. What is CNN?
2. Deep Neural Network vs Convolutional Neural Network
3. What is Kernel in CNN?
4. Understanding CNN architecture
5. What is the use of CNN?
6. What is Pooling and Padding?
7. What is the use of Pooling and Padding?
8. What is Convolution?
9. What is Tensorflow?
10. What is a feature map?
11. What is Data Augmentation?
12. How to load the data using tensorflow?
13. How to build a CNN model using tensorflow?
14. How to do data preprocessing?
15. How to build an API using Flask?
16. How to do real time prediction using gunicorn platform?


## Replicate 
### 1. Create a new environment

   - `conda create -p venv python==3.10 -y`
   - `conda activate venv/`

### 2. Install all the requirements

   - `python -m pip install --upgrade pip`
   - `python -m pip install -r requirements.txt`

### 3. Code Execution

   - Run `python engine.py` to train/predict/deploy 