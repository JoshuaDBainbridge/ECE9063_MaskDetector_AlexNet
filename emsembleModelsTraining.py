import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import layers, models
from tensorflow.keras.applications.mobilenet import MobileNet
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

import glob
import os
import random
import cv2

print("Declare Directory")
train_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Train"
val_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Validation"
test_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Test"
model_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/ALEXNET/Models/Final"

classes = ["With Mask", "Without Mask"]

print("Dataset loader")
train_datagen = ImageDataGenerator(
                                rescale=1./255,
                                rotation_range=0.2,
                                #width_shift_range=0.1,
                                #height_shift_range=0.1,
                                shear_range=0.2,
                                #zoom_range=0.09,
                                horizontal_flip=True,
                                vertical_flip=False,
                                #validation_split=0.1
                                )
val_datagen = ImageDataGenerator(rescale=1./255)
print("Image Generator Config")
target_size = (150, 150)
batch_size = 16

print("Load Dataset")
train_dataset = train_datagen.flow_from_directory(train_dir,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  shuffle=True)
val_dataset = val_datagen.flow_from_directory(val_dir,
                                              target_size=target_size,
                                              batch_size=batch_size,
                                              class_mode="categorical",
                                              shuffle=False)

# Build Model
num_classes = 2 # WithMask, WithoutMask
input_shape = (150, 150, 3)
load_dir = model_dir + "/Model_1"
model = keras.models.load_model(load_dir)
model.summary()

print("Setting backprop of model (how this model learning)")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
print("Training \n Training data set length", len(train_dataset))
EPOCHS = 50

history = model.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)

print("Review Our Model")
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(14,5))
grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
fig.add_subplot(grid[0])
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history["val_accuracy"], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

fig.add_subplot(grid[1])
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
print("Load Test Dataset")
test_dataset = val_datagen.flow_from_directory(test_dir,
                                            target_size=target_size,
                                            batch_size=1,
                                            class_mode=None,
                                            shuffle=False)

probabilities = model.predict_generator(test_dataset)
y_pred = probabilities.argmax(axis=-1)
y_test = test_dataset.classes
print("Accuracy Score of Model:", accuracy_score(y_pred,y_test))
print(classification_report(y_test, y_pred, target_names=classes))
