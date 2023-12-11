import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras import callbacks,optimizers
import h5py, io, zipfile

buffer = bytes()

zip_part_one = open('model.part01.rar', 'rb')
zip_part_two = open('model.part02.rar', 'rb')
zip_part_three = open('model.part03.rar', 'rb')
zip_part_four = open('model.part04.rar', 'rb')
zip_part_five = open('model.part05.rar', 'rb')
zip_part_six = open('model.part06.rar', 'rb')
zip_part_seven = open('model.part07.rar', 'rb')
zip_part_eight = open('model.part08.rar', 'rb')
zip_part_nine = open('model.part09.rar', 'rb')
zip_part_ten = open('model.part10.rar', 'rb')
zip_part_eleven = open('model.part11.rar', 'rb')
zip_part_twelve = open('model.part12.rar', 'rb')
zip_part_thirteen = open('model.part13.rar', 'rb')
zip_part_Fourteen = open('model.part14.rar', 'rb')
zip_part_Fifteen = open('model.part15.rar', 'rb')
zip_part_sixteen = open('model.part16.rar', 'rb')
zip_part_Seventeen = open('model.part17.rar', 'rb')
zip_part_eighteen = open('model.part18.rar', 'rb')
zip_part_nineteen = open('model.part19.rar', 'rb')
zip_part_twenty = open('model.part20.rar', 'rb')
zip_part_twentyone = open('model.part21.rar', 'rb')
zip_part_twentytwo = open('model.part22.rar', 'rb')
zip_part_twentythree = open('model.part23.rar', 'rb')
zip_part_twentyfour = open('model.part24.rar', 'rb')
zip_part_twentyfive = open('model.part25.rar', 'rb')
zip_part_twentysix = open('model.part26.rar', 'rb')
zip_part_twentyseven = open('model.part27.rar', 'rb')
zip_part_twentyeight = open('model.part28.rar', 'rb')
zip_part_twentynine = open('model.part29.rar', 'rb')
zip_part_thirty = open('model.part30.rar', 'rb')
zip_part_thirtyone = open('model.part31.rar', 'rb')
zip_part_thirtytwo = open('model.part32.rar', 'rb')
zip_part_thirtythree = open('model.part33.rar', 'rb')
zip_part_thirtyfour = open('model.part34.rar', 'rb')
zip_part_thirtyfive = open('model.part35.rar', 'rb')
zip_part_thirtysix = open('model.part36.rar', 'rb')
zip_part_thirtyseven = open('model.part37.rar', 'rb')
zip_part_thirtyeight = open('model.part38.rar', 'rb')
zip_part_thirtynine = open('model.part39.rar', 'rb')
zip_part_fourty = open('model.part40.rar', 'rb')
zip_part_fourtyone = open('model.part41.rar', 'rb')
zip_part_fourtytwo = open('model.part42.rar', 'rb')



buffer += zip_part_one.read()
buffer += zip_part_two.read()
buffer += zip_part_three.read()
buffer += zip_part_four.read()
buffer += zip_part_five.read()
buffer += zip_part_six.read()
buffer += zip_part_seven()
buffer += zip_part_eight.read()
buffer += zip_part_nine.read()
buffer += zip_part_ten.read()
buffer += zip_part_eleven.read()
buffer += zip_part_twelve.read()
buffer += zip_part_thirteen.read()
buffer += zip_part_fourteen.read()
buffer += zip_part_fifteen.read()
buffer += zip_part_sixteen.read()
buffer += zip_part_seventeen.read()
buffer += zip_part_eighteen.read()
buffer += zip_part_nineteen.read()
buffer += zip_part_twenty.read()
buffer += zip_part_twentyone.read()
buffer += zip_part_twentytwo.read()
buffer += zip_part_twentythree.read()
buffer += zip_part_twentyfour.read()
buffer += zip_part_twentyfive.read()
buffer += zip_part_twentysix.read()
buffer += zip_part_twentyseven.read()
buffer += zip_part_twentyeight.read()
buffer += zip_part_twentynine.read()
buffer += zip_part_thirty.read()
buffer += zip_part_thirtyone.read()
buffer += zip_part_thirtytwo.read()
buffer += zip_part_thirtythree.read()
buffer += zip_part_thirtyfour.read()
buffer += zip_part_thirtyfive.read()
buffer += zip_part_thirtysix.read()
buffer += zip_part_thirtyseven.read()
buffer += zip_part_thirtyeight.read()
buffer += zip_part_thirtynine.read()
buffer += zip_part_fourty.read()
buffer += zip_part_fourtyone.read()
buffer += zip_part_fourtytwo.read()


zf = zipfile.ZipFile(io.BytesIO(buffer))

f = io.BytesIO(zf.read('model.h5'))

zf = zipfile.ZipFile(io.BytesIO(buffer))

f = io.BytesIO(zf.read('model.h5'))

@st.cache(allow_output_mutation=True)
def load_model():
    h5 = h5py.File(f)
    return tf.keras.models.load_model(h5)

st.write("""
         # Item Purchase
         """
        )

file = st.file_uploader("Choose item images from computer", type=["jpg", "png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    size = (224, 224)    
    image_object = ImageOps.fit(image_data, size, Image.LANCZOS)
    image_array = np.asarray(image_object)
    image_cv = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_reshape = image_cv[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, load_model())
    class_names = ['Vegetables','Packages', 'Fruits']
    string="This image is: "+class_names[np.argmax(predictions)]
    st.success(string)
