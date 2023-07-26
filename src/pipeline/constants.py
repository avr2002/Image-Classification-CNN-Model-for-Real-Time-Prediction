from pathlib import Path

TRAIN_DATA_DIR = Path("./input/train_data/")
TEST_DATA_DIR = Path("./input/test_data/")

CLASS_NAMES = ['driving_license', 'others', 'social_security']

BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = 180, 180
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # 3 color channels
SEED = 42

VALIDATION_SPLIT=0.2
EPOCHS = 20

MODEL_DIR = Path("./output/saved_model/")