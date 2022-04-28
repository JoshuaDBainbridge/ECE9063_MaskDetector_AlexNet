import matplotlib.pyplot as plt
import os
import random
import cv2

print("Declare Directory")
train_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Train"
val_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Validation"
test_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/Project/maskData/FaceMaskDataset/Test"
n = 10000
for i in range(n):
    sample = random.choice(os.listdir(train_dir + "/WithMask"))
    #sample = random.choice(os.listdir(train_dir + "/WitouthMask"))
    img_dir = train_dir + "/WithMask/" + sample
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xlabel("With Mask")
    plt.show()
    val = input("Would you like to keep? ")
    if val == 'n':
        os.remove("img_dir")



