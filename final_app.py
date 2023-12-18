import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

pip install opencv-python-headless

model = tf.keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/GroceryStoreDataset-master/Item.hdf5")
### load file
uploaded_file = st.file_uploader("Choose an image file", type="jpg")

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=[ ("Apple", "Avocado", "Orange", "Pineapple", "Juice", "Milk", "Yoghurt", "Cabbage", "Carrot", "Potato", "Tomato" ]
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
