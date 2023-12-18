import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
import numpy as np
from PIL import Image, ImageOps
import requests
import io

# Direct link to the raw model file on GitHub
MODEL_URL = "https://github.com/ZjCantarona/Emtech2Finals/raw/main/Item.hdf5"

# Load the model
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

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

map_dict = {
    0: 'Apple',
    1: 'Avocado',
    2: 'Orange', 
    3: 'Pineapple',
    4: 'Juice', 
    5: 'Milk', 
    6: 'Yoghurt',
    7: 'Cabbage', 
    8: 'Carrot', 
    9: 'Potato', 
    10:'Tomato',
        }


if uploaded_file is not None:
    # Read the image file
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Resize the image to the required dimensions
    resized = cv2.resize(img, (224, 224))

    # Preprocess the image
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized["Fruits", "Vegetables", "Packages"]

    Generate_pred = st.button("Generate Prediction")

    if Generate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict[prediction]))
