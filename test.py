from model import Tracker_Model
from image import Dataset
from visualizer import plot
import os


height = 1080 // 4
width = 1920 // 4
# Load model
NN = Tracker_Model(dummy = True, image_dim = (height, width, 3))
NN.summary()

# Load dataset
BASE_PATH = os.path.dirname(__file__)
WEIGHT_PATH = os.path.join(BASE_PATH, './training/ckpt.weights.h5')
TEST_PATH = os.path.join(BASE_PATH, './Soccerball/test')
CSV_PATH = os.path.join(TEST_PATH, '_annotations.csv')

dataset = Dataset(TEST_PATH, CSV_PATH, image_size = (height, width))
dataset = dataset.load()

# Build model before loading

_ = NN(next(iter(dataset.take(1)))[0])
NN.load_weights(WEIGHT_PATH)

# Evaluate model
loss, acc = NN.evaluate(dataset)

print(f'Loss : {loss:.2f}')
print(f'Accuracy : {acc:.2f}')

for batch in dataset:
    images, labels = batch

    plot(images, labels, num_samples=4)




