import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# Load data

batch_size = 32

train_data = ImageDataGenerator().flow_from_directory(directory = 'FER-2013/train', 
                                                      target_size = (48,48),
                                                      color_mode = 'grayscale',
                                                      batch_size = batch_size,
                                                      shuffle = True,
                                                      class_mode='categorical')

test_data = ImageDataGenerator().flow_from_directory(directory = 'FER-2013/test', 
                                                      target_size = (48,48),
                                                      color_mode = 'grayscale',
                                                      batch_size = batch_size,
                                                      shuffle = False,
                                                      class_mode='categorical')


# Model

learning_rate = 0.0001

model = keras.Sequential()

# Conv layer 1
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding= 'same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Conv layer 2
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding= 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Conv layer 3
model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding= 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten)

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7))

model.compile(optimizer = Adam(learning_rate=learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# Prepare for training

checkpoint = ModelCheckpoint("models/emotion_model.h5", monitor='')