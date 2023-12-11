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

        # Display the reshaped image
        st.image(img_reshape, channels="RGB", use_column_width=True)

        prediction = model.predict(img_reshape)
        print("Raw Predictions:", prediction)

        class_names = ['Vegetables', 'Packages', 'Fruits']
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]

        print("Predicted Class:", predicted_class_name)

        # Threshold for displaying predictions
        threshold = 0.5
        if np.max(prediction) > threshold:
            string = "This image is: " + predicted_class_name
            st.success(string)
        else:
            st.warning("Model confidence is below the threshold.")

        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # Check if the model is loaded successfully
        model = load_model()
        if model is not None:
            predictions = import_and_predict(image, model)
            if predictions is not None:
                # Additional code, if needed, can be added here
                pass
    except Exception as e:
        st.error(f"Unexpected error: {e}")
