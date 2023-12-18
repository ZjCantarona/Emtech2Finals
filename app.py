import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

def load_model():
    model = tf.keras.models.load_model('Item.hdf5')
    return model

model = load_model()

st.write("""
# Item Purchase of Group 2
""")
file = st.file_uploader("Choose plant photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Vegetables', 'Packages', 'Fruits']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)

