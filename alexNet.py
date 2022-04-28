import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

import zipfile
import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

from tqdm import tqdm

print("Loaded Tensorflow")

# DATA_PATH ='C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Train/WithMask.zip'
# zip_object = zipfile.ZipFile(DATA_PATH, mode='r')  # https://docs.python.org/3/library/zipfile.html#zipfile-objects
# zip_object.extractall('./')
# zip_object.close

DATA_TRAIN_SET = 'C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Train'
DATA_TEST_SET = 'C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Test'
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200


def create_dataset(DATA_PATH):
    img_data_array = []
    class_name = []

    for directory in os.listdir(DATA_PATH):
        print(directory)
        for file in os.listdir(os.path.join(DATA_PATH, directory)):
            image_path = os.path.join(DATA_PATH, directory, file)
            image = cv.imread(image_path, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(directory)
    return img_data_array, class_name


print("Extracting Training Mask Data")
DATA_IMAGES_TRAIN, DATA_LABELS_TRAIN = create_dataset(DATA_TRAIN_SET)

print("Extracting Training No Mask Data")
DATA_IMAGES_TEST, DATA_LABELS_TEST = create_dataset(DATA_TEST_SET)

print("TRAIN DATA LOADED: ", len(DATA_LABELS_TRAIN))
print("TEST DATA LOADED: ", len(DATA_LABELS_TEST))

X_train = DATA_IMAGES_TRAIN
X_test = DATA_IMAGES_TEST
y_train = DATA_LABELS_TRAIN
y_test = DATA_LABELS_TEST

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images[:1000]
training_labels = training_labels[:1000]
test_images = test_images[:100]
test_labels = test_labels[:100]

training_images = tf.map_fn(lambda i: tf.stack([i]*3, axis=-1), training_images).numpy()
test_images = tf.map_fn(lambda i: tf.stack([i]*3, axis=-1), test_images).numpy()

training_images = tf.image.resize(training_images, [224, 224]).numpy()
test_images = tf.image.resize(test_images, [224, 224]).numpy()

training_images = training_images.reshape(1000, 224, 224, 3)
training_images = training_images / 255.0
test_images = test_images.reshape(100, 224, 224, 3)
test_images = test_images / 255.0

training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

num_len_train = int(0.8 * len(training_images))

ttraining_images = training_images[:num_len_train]
ttraining_labels = training_labels[:num_len_train]

valid_images = training_images[num_len_train:]
valid_labels = training_labels[num_len_train:]

training_images = ttraining_images
training_labels = ttraining_labels

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                           input_shape=(224, 224, 3)),

    tf.keras.layers.MaxPooling2D(3, strides=2),

    tf.keras.layers.Conv2D(256, (5, 5), activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), bias_initializer='ones'),

    tf.keras.layers.MaxPooling2D(3, strides=2),

    tf.keras.layers.Conv2D(384, (3, 3), activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)),

    tf.keras.layers.Conv2D(384, (3, 3), activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                           bias_initializer='ones'),

    tf.keras.layers.Conv2D(384, (3, 3), activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), \
                           bias_initializer='ones'),

    tf.keras.layers.MaxPooling2D(3, strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096, kernel_initializer= tf.random_normal_initializer(mean=0.0, stddev=0.01), bias_initializer='ones'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(4096, kernel_initializer= tf.random_normal_initializer(mean=0.0, stddev=0.01), bias_initializer='ones'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer= tf.random_normal_initializer(mean=0.0, stddev=0.01))
])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(5)])
# printing the model
print(model.summary())
# Learning Rate reduces on plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001)
# fit the model to the training set & and evaluate against the validation data
model.fit(training_images, training_labels, batch_size=128, validation_data=(valid_images, valid_labels), epochs=10, callbacks=[reduce_lr])
