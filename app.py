import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
import numpy as np
from PIL import Image, ImageOps
import cv2

# Define the path to your model file
MODEL_PATH = "path/to/your/model.h5"

@st.cache(allow_output_mutation=True)
def load_model():
    # Load the model using tf.keras.models.load_model
    return tf.keras.models.load_model(MODEL_PATH)

st.write("""
         # Item Purchase
         """
        )

file = st.file_uploader("Choose item images from computer", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    size = (224, 224)
    image_object = ImageOps.fit(image_data, size, Image.LANCZOS)
    image_array = np.asarray(image_object)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_reshape = image_cv[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Check if the model is loaded successfully
    model = load_model()
    if model is not None:
        predictions = import_and_predict(image, model)
        class_names = ['Vegetables', 'Packages', 'Fruits']
        string = "This image is: " + class_names[np.argmax(predictions)]
        st.success(string)
    else:
        st.text("Error loading the model. Please check the model path.")
