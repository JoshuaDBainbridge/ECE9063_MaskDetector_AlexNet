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
save_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/ALEXNET/Models/TrainedFinal"
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

#full = [[16, 16, 16, 16, 16], [32, 32, 32, 32, 32], [64, 64, 64, 64, 64], [96, 96, 96, 96, 96], [128, 128, 128, 128, 128], [16, 32, 64, 96, 128], [128, 96, 64, 32, 16], [16, 64, 128, 96, 32]]
full = [[16, 16, 16], [32, 32, 32], [64, 64, 64], [96, 96, 96], [128, 128, 128], [16, 32, 64], [16, 64, 96], [16, 64, 128], [32, 64, 96], [32, 64, 128], [64, 96, 128]]
# MODEL 1
load_dir = model_dir + "/Model_1"
model_1 = keras.models.load_model(load_dir)
model_1.summary()
model_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_1.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)
model_1.save(save_dir + "/Model_1")
#MODEL 2
load_dir = model_dir + "/Model_2"
model_2 = keras.models.load_model(load_dir)
model_2.summary()
model_2.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_2.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)
model_2.save(save_dir + "/Model_2")
#MODEL 3
load_dir = model_dir + "/Model_3"
model_3 = keras.models.load_model(load_dir)
model_3.summary()
model_3.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_3.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)
model_3.save(save_dir + "/Model_3")
#MODEL 4
load_dir = model_dir + "/Model_4"
model_4 = keras.models.load_model(load_dir)
model_4.summary()
model_4.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_4.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)
model_4.save(save_dir + "/Model_4")
#MODEL 5
load_dir = model_dir + "/Model_5"
model_5 = keras.models.load_model(load_dir)
model_5.summary()
model_5.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_5.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)
model_5.save(save_dir + "/Model_5")
#MODEL 6
load_dir = model_dir + "/Model_6"
model_6 = keras.models.load_model(load_dir)
model_6.summary()
model_6.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_6.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)

model_6.save(save_dir + "/Model_6")
#MODEL 7
load_dir = model_dir + "/Model_7"
model_7 = keras.models.load_model(load_dir)
model_7.summary()
model_7.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_7.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)

model_7.save(save_dir + "/Model_7")
#MODEL 8
load_dir = model_dir + "/Model_8"
model_8 = keras.models.load_model(load_dir)
model_8.summary()
model_8.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_8.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)
model_8.save(save_dir + "/Model_8")
#MODEL 9
load_dir = model_dir + "/Model_9"
model_9 = keras.models.load_model(load_dir)
model_9.summary()
model_9.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_9.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)
model_9.save(save_dir + "/Model_9")
#MODEL 10
load_dir = model_dir + "/Model_10"
model_10 = keras.models.load_model(load_dir)
model_10.summary()
model_10.compile(optimizer="adam", loss="categorical_crossentropy", metrics ="accuracy")
EPOCHS = 50
history = model_10.fit_generator(train_dataset,
                               steps_per_epoch=len(train_dataset)//train_dataset.batch_size,
                               validation_data=val_dataset,
                               validation_steps=len(val_dataset)//val_dataset.batch_size,
                               epochs=EPOCHS)
model_10.save(save_dir + "/Model_10")
print("Load Test Dataset")
test_dataset = val_datagen.flow_from_directory(test_dir,
                                            target_size=target_size,
                                            batch_size=1,
                                            class_mode=None,
                                            shuffle=False)


def preprocessing_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    #plt.figure()
    #plt.imshow(img/255)
    #plt.colorbar()
    #plt.grid(False)
    #plt.show()
    return img

results = [[0, 0],[0, 0]]
ensemble = [1]*10
j = 6
w = 5
num = 0
plt.figure(figsize=(15, 15))
for x in range(w):
    for i in range(j):
        #random_test_img = random.choice(glob.glob(test_dir+"/WithMask/79_new.png"))
        random_test_img = random.choice(glob.glob(test_dir + "/WithMask/*"))
        print(random_test_img)
        img_test = cv2.imread(random_test_img)
        img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
        img_test_pro = preprocessing_img(img_test)

        #MODEL 1
        result = model_1.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[0] *= score
        else:
            results[1][1] += 1
            ensemble[0] *= -score
        #MODEL 2
        result = model_2.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[1] *= score
        else:
            results[1][1] += 1
            ensemble[1] *= -score
        #MODEL 3
        result = model_3.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[2] *= score
        else:
            results[1][1] += 1
            ensemble[2] *= -score
        #MODEL 4
        result = model_4.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[3] *= score
        else:
            results[1][1] += 1
            ensemble[3] *= -score
        #MODEL 5
        result = model_5.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[4] *= score
        else:
            results[1][1] += 1
            ensemble[4] *= -score
        #MODEL 6
        result = model_6.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[5] *= score
        else:
            results[1][1] += 1
            ensemble[5] *= -score
        #MODEL 7
        result = model_7.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[6] *= score
        else:
            results[1][1] += 1
            ensemble[6] *= -score
        #MODEL 8
        result = model_8.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[7] *= score
        else:
            results[1][1] += 1
            ensemble[7] *= -score
        #MODEL 9
        result = model_9.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[8] *= score
        else:
            results[1][1] += 1
            ensemble[8] *= -score
        #MODEL 10
        result = model_10.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "With Mask":
            results[0][0] += 1
            ensemble[9] *= score
        else:
            results[1][1] += 1
            ensemble[9] *= -score

        finalScore = sum(ensemble)/10
        if finalScore < 0:
            predicted_class = "Without Mask"
        elif finalScore > 0:
            predicted_class = "With Mask"

        printScore = str(finalScore)
        plt.subplot(w, j, 1 + num)
        plt.tight_layout
        plt.imshow(img_test)
        plt.title(predicted_class + "Con: " + "{:.2f}".format(finalScore))
        num += 1
plt.show()


j = 6
w = 5
num = 0
ensemble2 = [1]*10
plt.figure(figsize=(15, 15))
for x in range(w):
    for i in range(j):
        random_test_img = random.choice(glob.glob(test_dir+"/WithoutMask/*"))
        print(random_test_img)
        img_test = cv2.imread(random_test_img)
        img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
        img_test_pro = preprocessing_img(img_test)

        # MODEL 1
        result = model_1.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[0] *= score
        else:
            results[0][1] += 1
            ensemble2[0] *= -score

        # MODEL 2
        result = model_2.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[1] *= score
        else:
            results[0][1] += 1
            ensemble2[1] *= -score

        # MODEL 3
        result = model_3.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[2] *= score
        else:
            results[0][1] += 1
            ensemble2[2] *= -score

        # MODEL 4
        result = model_4.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[3] *= score
        else:
            results[0][1] += 1
            ensemble2[3] *= -score

        # MODEL 5
        result = model_5.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[4] *= score
        else:
            results[0][1] += 1
            ensemble2[4] *= -score

        # MODEL 6
        result = model_6.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[5] *= score
        else:
            results[0][1] += 1
            ensemble2[5] *= -score

        # MODEL 7
        result = model_7.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[6] *= score
        else:
            results[0][1] += 1
            ensemble2[6] *= -score

        # MODEL 8
        result = model_8.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[7] *= score
        else:
            results[0][1] += 1
            ensemble2[7] *= -score
        # MODEL 9
        result = model_9.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[8] *= score
        else:
            results[0][1] += 1
            ensemble2[8] *= -score

        # MODEL 10
        result = model_10.predict(img_test_pro)
        score = np.max(result)
        predicted_class = classes[np.argmax(result)]
        if predicted_class == "Without Mask":
            results[1][0] += 1
            ensemble2[9] *= score
        else:
            results[0][1] += 1
            ensemble2[9] *= -score

        finalScore = sum(ensemble)/10
        if finalScore < 0:
            predicted_class = "Without Mask"
        elif finalScore > 0:
            predicted_class = "With Mask"

        printScore = str(finalScore)
        plt.subplot(w, j, 1 + num)
        plt.tight_layout
        # plt.suptitle("Confident: " + printScore)
        plt.imshow(img_test)
        plt.title(predicted_class +"Con: " + "{:.2f}".format(score))
        #plt.xlabel("Confident: " + printScore)
        num += 1
plt.show()

print("WM, True: ", results[0][0])
print("WM, False: ", results[0][1])
print("WOM, True: ", results[1][0])
print("WOM, False: ", results[1][1])




