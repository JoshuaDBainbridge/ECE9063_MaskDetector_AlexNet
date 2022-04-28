from keras import layers, models
from tensorflow import keras
import os
save_dir = "C:/Users/jdbai/Documents/GraduateSchool/Term3/ECE9063/ALEXNET/Models/Layers3"
# Build Model
num_classes = 2  # WithMask, WithoutMask
input_shape = (150, 150, 3)

number_of_perceptrions = [16, 32, 64, 96, 128]
number_of_layers = [3, 5, 7]
#full = [[16, 16, 16, 16, 16], [32, 32, 32, 32, 32], [64, 64, 64, 64, 64], [96, 96, 96, 96, 96], [128, 128, 128, 128, 128], [16, 32, 64, 96, 128], [128, 96, 64, 32, 16], [16, 64, 128, 96, 32]]
full = [[16, 16, 16], [32, 32, 32], [64, 64, 64], [96, 96, 96], [128, 128, 128], [16, 32, 64], [16, 64, 96], [16, 64, 128], [32, 64, 96], [32, 64, 128], [64, 96, 128]]
types_of_structure = ['Constant', 'Accending', 'Decending', 'MiddleOut']
activation = ['relu', 'sigmoid', 'softmax']
x = len(activation) * len(full)


for i in range(len(full)):
    for j in range(len(activation)):
        model_dir = save_dir + "/Model_"+str(i)+"_"+activation[j]
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model = models.Sequential()
        # 1st Conv layer
        #model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(layers.Conv2D(full[i][0], (3, 3), activation=activation[j], padding='same', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        # 2nd Conv layer
        #model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(full[i][1], (3, 3), activation=activation[j], padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 3rd Conv layer
        #model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(full[i][2], (3, 3), activation=activation[j], padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 4th Conv layer
        ##model.add(layers.Conv2D(96, (3, 3), activation='relu', padding='same'))
        #model.add(layers.Conv2D(full[i][3], (3, 3), activation=activation[j], padding='same'))
        #model.add(layers.MaxPooling2D((2, 2)))
        # 5th Conv layer
        ##model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        #model.add(layers.Conv2D(full[i][4], (3, 3), activation=activation[j], padding='same'))
        #model.add(layers.MaxPooling2D((2, 2)))
        # FC layers
        model.add(layers.Flatten())
        #model.add(layers.Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(layers.Dense(1024))
        #model.add(layers.Dropout(0.2))
        #model.add(layers.Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(layers.Dense(64))
        #model.add(layers.Dropout(0.2))

        model.add(layers.Dense(num_classes, activation='softmax'))

        model.summary()
        print("SAVEING")
        model.save(model_dir)
        history = model.fit_generator(train_dataset,
                                     steps_per_epoch=len(train_dataset) // train_dataset.batch_size,
                                     validation_data=val_dataset,
                                     validation_steps=len(val_dataset) // val_dataset.batch_size,
                                     epochs=EPOCHS)

        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(14, 5))
        grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
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


