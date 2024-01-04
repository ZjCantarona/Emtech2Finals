import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
import numpy as np
from PIL import Image, ImageOps
import requests
import io

# Direct link to the raw model file on GitHub
MODEL_URL = "https://github.com/ZjCantarona/Emtech2Finals/blob/main/Final.h5"
# Load the model


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
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

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

map_dict = {
    "Fruit", "Packages","Vegetables",}



if uploaded_file is not None:
    # Read the image file using PIL
    image_pil = Image.open(uploaded_file)

    # Resize the image to the required dimensions
    size = (224, 224)
    image_resized = ImageOps.fit(image_pil, size, Image.LANCZOS)

    # Preprocess the image
    image_array = np.asarray(image_resized)
    img_reshape = image_array[np.newaxis, ...]
    img_preprocessed = mobilenet_v2_preprocess_input(img_reshape)

    # Load the model
    model = load_model()

    Generate_pred = st.button("Generate Prediction")

    if Generate_pred and model is not None:
        # Get the prediction probabilities for each class
        predictions = model.predict(img_preprocessed)[0]

        # Display the predicted class probabilities
        for i, prob in enumerate(predictions):
            st.text(f"Probability for {map_dict}")
