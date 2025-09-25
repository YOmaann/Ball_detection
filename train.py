from model import Tracker_Model
from image import Dataset
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import math


# configurations
width = 1920 // 4
height = 1080 // 4
# Load model


NN = Tracker_Model(image_dim = (height, width, 3), dummy = False)
NN.summary()

# Load dataset
BASE_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_PATH, './Soccerball/train')
VALID_PATH = os.path.join(BASE_PATH, './Soccerball/valid')
CSV_PATH = os.path.join(TRAIN_PATH, '_annotations.csv')
V_CSV = os.path.join(VALID_PATH, '_annotations.csv')

_dataset = Dataset(TRAIN_PATH, CSV_PATH, image_size = ( height, width))
dataset = _dataset.load().repeat()


# Use validation set
# training_ds, validation_ds = tf.keras.utils.split_dataset(dataset, left_size = 0.8, shuffle = True)

_validation_dataset = Dataset(VALID_PATH, V_CSV, image_size = (height, width))
validation_ds = _validation_dataset.load().repeat()

checkpoint = ModelCheckpoint(filepath="training/ckpt.weights.h5",
                             save_weights_only=True,
                             verbose=1)

steps_per_epoch = math.ceil(_dataset.size // _dataset.batch_size)
validation_steps = math.ceil(_validation_dataset.size // _validation_dataset.batch_size)

NN.fit(dataset, epochs = 30, validation_data = validation_ds, callbacks = [checkpoint], steps_per_epoch = steps_per_epoch, validation_steps = validation_steps)


