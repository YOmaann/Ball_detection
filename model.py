# This file contains a simple model for this task.

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.random import normal
import os
from metric import iou_loss, l1_loss


class Tracker_Model(Model):
    def __init__(self, image_dim = (244, 244, 3), metrics = ['mae'], dummy = False):
        super(Tracker_Model, self).__init__()

        # Add convolution layer 1
        self.conv1 = Conv2D(32, (3, 3), activation = 'relu')
        self.bn1 = BatchNormalization()
        # Pooling
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)
        #Add convolution layer 2
        self.conv2 = Conv2D(64, (3, 3), activation = 'relu')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)
        # Add convolution layer 3
        self.conv3 = Conv2D(128, (3, 3), activation = 'relu')
        # self.bn3 = BatchNormalization()
        # Another convolution layer
        # self.conv4 = Conv2D(128, (7, 7), activation = 'relu')
        # self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=1)


        # Feeding the output to a DNN
        self.flatten = Flatten()
        # self.gap = GlobalAveragePooling2D()
        # Layer with 100 Nurons
        self.fc1 = Dense(256, activation = 'relu')

        # Testing with dropout layer
        self.drop1 = Dropout(0.4)
        # self.bn4 = BatchNormalization()
        # Layer w 100 Neurons
        self.fc2 = Dense(100, activation = 'relu')
        self.drop2 = Dropout(0.4)
        # self.bn5 = BatchNormalization()
        # Output Layer with 4 Nurons for x_min, y_min, x_max, y_max
        self.out = Dense(4, activation = 'linear')


        
        self.compile(optimizer="adam", loss=[l1_loss], metrics = metrics)

        if dummy:
            print('Adding dummy value')
            dummy_input = normal((1, *image_dim))
            _ = self(dummy_input)

        # Path to save the model in
        # self.ch_path = ch_path
        # self.ch_dir = os.path.dirname(ch_path)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.pool3(x)
        # x = self.conv4(x)
        # x = self.flatten(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x)
        # x = self.bn4(x)
        x = self.fc2(x)
        x = self.drop2(x)
        # x = self.bn5(x)
        return self.out(x)

    # def fit(self, *args, **kwargs):
    #     # Add the model saving point automatically
    #     callbacks = kwargs.get("callbacks", [])

    #     callbacks.append(ModelCheckpoint(filepath=self.ch_path,
    #                                              save_weights_only=True,
    #                                              verbose=1))

    #     kwargs['callbacks'] = callbacks
    #     super().fit(*args, **kwargs)
