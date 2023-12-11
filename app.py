import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras import callbacks,optimizers
import h5py, io, zipfile

buffer = bytes()

zip_part_one = open('vgg16-model.part01.rar', 'rb')
zip_part_two = open('vgg16-model.part02.rar', 'rb')
zip_part_three = open('vgg16-model.part03.rar', 'rb')
zip_part_four = open('vgg16-model.part04.rar', 'rb')
zip_part_five = open('vgg16-model.part05.rar', 'rb')
zip_part_six = open('vgg16-model.part06.rar', 'rb')
zip_part_seven = open('vgg16-model.part07.rar', 'rb')
zip_part_eight = open('vgg16-model.part08.rar', 'rb')
zip_part_nine = open('vgg16-model.part09.rar', 'rb')
zip_part_ten = open('vgg16-model.part10.rar', 'rb')
zip_part_eleven = open('vgg16-model.part11.rar', 'rb')
zip_part_twelve = open('vgg16-model.part12.rar', 'rb')
zip_part_thirteen = open('vgg16-model.part13.rar', 'rb')
zip_part_Fourteen = open('vgg16-model.part14.rar', 'rb')
zip_part_Fifteen = open('vgg16-model.part15.rar', 'rb')
zip_part_sixteen = open('vgg16-model.part16.rar', 'rb')
zip_part_Seventeen = open('vgg16-model.part17.rar', 'rb')
zip_part_eighteen = open('vgg16-model.part18.rar', 'rb')
zip_part_nineteen = open('vgg16-model.part19.rar', 'rb')
zip_part_twenty = open('vgg16-model.part20.rar', 'rb')
zip_part_twentyone = open('vgg16-model.part21.rar', 'rb')
zip_part_twentytwo = open('vgg16-model.part22.rar', 'rb')
zip_part_twentythree = open('vgg16-model.part23.rar', 'rb')
zip_part_twentyfour = open('vgg16-model.part24.rar', 'rb')
zip_part_twentyfive = open('vgg16-model.part25.rar', 'rb')
zip_part_twentysix = open('vgg16-model.part26.rar', 'rb')
zip_part_twentyseven = open('vgg16-model.part27.rar', 'rb')
zip_part_twentyeight = open('vgg16-model.part28.rar', 'rb')
zip_part_twentynine = open('vgg16-model.part29.rar', 'rb')
zip_part_thirty = open('vgg16-model.part30.rar', 'rb')
zip_part_thirtyone = open('vgg16-model.part31.rar', 'rb')
zip_part_thirtytwo = open('vgg16-model.part32.rar', 'rb')
zip_part_thirtythree = open('vgg16-model.part33.rar', 'rb')
zip_part_thirtyfour = open('vgg16-model.part34.rar', 'rb')
zip_part_thirtyfive = open('vgg16-model.part35.rar', 'rb')
zip_part_thirtysix = open('vgg16-model.part36.rar', 'rb')
zip_part_thirtyseven = open('vgg16-model.part37.rar', 'rb')
zip_part_thirtyeight = open('vgg16-model.part38.rar', 'rb')
zip_part_thirtynine = open('vgg16-model.part39.rar', 'rb')
zip_part_fourty = open('vgg16-model.part40.rar', 'rb')
zip_part_fourtyone = open('vgg16-model.part41.rar', 'rb')
zip_part_fourtytwo = open('vgg16-model.part42.rar', 'rb')



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

f = io.BytesIO(zf.read('vgg16-model.h5'))

zf = zipfile.ZipFile(io.BytesIO(buffer))

f = io.BytesIO(zf.read('vgg16-model.h5'))

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
