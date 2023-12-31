# -*- coding: utf-8 -*-
"""app

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gO2dOgFageGugKDym7kuhFHj9SMPKgPR
"""



!pip install streamlit
!pip install pyngrok
!pip install streamlit
!pip install opencv-python

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras import callbacks,optimizers
import numpy as np
from google.colab import drive

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/CPE 019 Final Exam/ItemPurchase/GroceryStoreDataset-master

path = '/content/drive/MyDrive/CPE 019 Final Exam/ItemPurchase/GroceryStoreDataset-master/sample_images/dataset/train'

file_list = []

for root, dirs, files in os.walk(path):
#     print('Root : ', root)
#     print('Dirs : ', dirs)
#     print('Files : ', files)
    for file in files:
        file_list.append(os.path.join(root, file))


# print(file_list)
print(len(file_list))

path = '/content/drive/MyDrive/CPE 019 Final Exam/ItemPurchase/GroceryStoreDataset-master/sample_images/dataset/iconic-images-and-descriptions/Fruit/Apple'

folder_list = []

for root, dirs, files in os.walk(path):
#     print('Root : ', root)
#     print('Dirs : ', dirs)
#     print('Files : ', files)

    if len(dirs) > 0:
        folder_list.append(dirs)


print(folder_list)
print(len(folder_list))

# Commented out IPython magic to ensure Python compatibility.

import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline

import glob


from tensorflow.keras.preprocessing import image_dataset_from_directory

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255., # Rescaling
    rotation_range = 40, # for augmentation
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

train_path = '/content/drive/MyDrive/CPE 019 Final Exam/ItemPurchase/GroceryStoreDataset-master/sample_images/dataset/train'
image_size = (224, 224)
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_path,
    batch_size = batch_size,
    class_mode = 'categorical',
    target_size = image_size
)

train_generator

train_generator[1]

train_generator[0][0][0]

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 2, figsize=(5,5))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


augmented_images = [train_generator[0][0][0] for i in range(10)]
plotImages(augmented_images)

labels = (train_generator.class_indices)
labels

labels = dict((v,k) for k,v in labels.items())
labels

no_classes = len(labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Create the model
model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Display a model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy']
             )

# Start training
model.fit(
        train_generator,
        epochs = 10,
        shuffle = False
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

test_path = '/content/drive/MyDrive/CPE 019 Final Exam/ItemPurchase/GroceryStoreDataset-master/sample_images/dataset/test'

test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    batch_size = batch_size,
    class_mode = 'categorical',
    target_size = image_size,
    shuffle = False,
    seed = 10

)

pred = model.predict_generator(
    test_generator,
    verbose=1
)

predicted_class_indices=np.argmax(pred,axis=1)

predictions = [labels[k] for k in predicted_class_indices]

predictions

augmented_images = [test_generator[0][0][0] for i in range(10)]
plotImages(augmented_images)

filenames = test_generator.filenames
results = pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

results.head()

results.tail()

from tensorflow.keras.models import load_model
model.save('/content/drive/MyDrive/CPE 019 Final Exam/ItemPurchase/GroceryStoreDataset-master/sample_images/Saved Models/model.h5')

