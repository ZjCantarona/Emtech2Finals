# -*- coding: utf-8 -*-
"""final_app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PUCUBHQOtLczM3O3ICARFgH5lGsJ-WRQ
"""

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/GroceryStoreDataset-master/Item.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'Fruit(Apple, Avocado, Orange, Pineapple)',
            1: 'Packages(Juice, Milk, Yoghurt)',
            2: 'Vegetable (Cabbage, Carrot, Potato, Tomato)',
            }

#resized = mobilenet_v2_preprocess_input(resized)
#    img_reshape = resized[np.newaxis,...]

Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
