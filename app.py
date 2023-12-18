import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
import numpy as np
from PIL import Image, ImageOps
import requests
import io  # Import the standard Python io module

# Direct link to the raw model file on GitHub
MODEL_URL = "https://github.com/ZjCantarona/Emtech2Finals/blob/main/Item.h5"

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

map_dict = {
    0: 'Fruit(Apple, Avocado, Orange, Pineapple)',
    1: 'Packages(Juice, Milk, Yoghurt)',
    2: 'Vegetable (Cabbage, Carrot, Potato, Tomato)',
}

if uploaded_file is not None:
    # Read the image file
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Resize the image to the required dimensions
    resized = cv2.resize(img, (224, 224))

    # Preprocess the image
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    Generate_pred = st.button("Generate Prediction")

    if Generate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict[prediction]))
