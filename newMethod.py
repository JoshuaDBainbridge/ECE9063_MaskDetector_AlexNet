import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import layers
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

classes = ["With Mask", "Without Mask"]

n = 5
print("Check Image")
plt.figure(figsize=(15, n))
for i in range(n):
    # read image
    sample = random.choice(os.listdir(train_dir + "/WithMask"))
    # print("filename:", sample)
    img_dir = train_dir + "/WithMask/" + sample
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plot image
    plt.subplot(1, n, 1+i)
    plt.imshow(img)
    plt.xlabel("With Mask")
plt.show()

plt.figure(figsize=(15, n))
for i in range(n):
    # read image
    sample = random.choice(os.listdir(train_dir + "/WithoutMask"))
    # print("filename:", sample)
    img_dir = train_dir + "/WithoutMask/" + sample
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plot image
    plt.subplot(1, n, 1+i)
    plt.imshow(img)
    plt.xlabel("Without Mask")
plt.show()

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
batch_size = 32

print("Load Dataset")
train_dataset = train_datagen.flow_from_directory(train_dir,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  shuffle=True)
# MY CODE
print(train_dataset)
# END MY CODE
val_dataset = val_datagen.flow_from_directory(val_dir,
                                              target_size=target_size,
                                              batch_size=batch_size,
                                              class_mode="categorical",
                                              shuffle=False)

base_model = tf.keras.applications.MobileNet(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
for layer in base_model.layers:
    layer.trainable = False
model = keras.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(2, activation="softmax"))

print("Setting backprop of model (how this model learning)")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
print("Training")
EPOCHS = 10
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
plt.plot(history.history['val_accuracy'], label='val accuracy')
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

def preprocessing_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    return img
random_test_img = random.choice(glob.glob(test_dir+"/*/*"))
print(random_test_img)
img_test = cv2.imread(random_test_img)
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
plt.imshow(img_test)
plt.show()

img_test = preprocessing_img(img_test)
result = model.predict(img_test)
score = np.max(result)
predicted_class = classes[np.argmax(result)]
print(predicted_class)
print("Confident: ", score)