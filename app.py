import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
import numpy as np
from PIL import Image, ImageOps
import requests
import io  # Import the standard Python io module

# Direct link to the raw model file on GitHub
MODEL_URL = "https://github.com/your-username/your-repo/raw/master/path/to/your/model.h5"

@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Fetch the model from GitHub using requests
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        
        # Load the model from the content of the response using io.BytesIO
        model_content = response.content
        return tf.keras.models.load_model(io.BytesIO(model_content))
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

st.write("""
         # Item Purchase
         """
        )

file = st.file_uploader("Choose item images from computer", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    try:
        size = (224, 224)
        image_object = ImageOps.fit(image_data, size, Image.LANCZOS)
        image_array = np.asarray(image_object)
        img_reshape = image_array[np.newaxis, ...]

        # Apply MobileNetV2 preprocessing
        img_preprocessed = mobilenet_v2_preprocess_input(img_reshape)

        # Display the original image
        st.image(image_object, channels="RGB", use_column_width=True)

        # Make prediction
        predictions = model.predict(img_preprocessed)
        class_names = ['Vegetables', 'Packages', 'Fruits']
        predicted_class = class_names[np.argmax(predictions)]

        # Display the prediction
        st.success(f"This image is: {predicted_class}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # Check if the model is loaded successfully
        model = load_model()
        if model is not None:
            import_and_predict(image, model)
    except Exception as e:
        st.error(f"Unexpected error: {e}")
