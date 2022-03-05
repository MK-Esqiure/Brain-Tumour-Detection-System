#Name: Kimingi Joshua Mukono
#Reg no: P15/1623/2019

import streamlit as st
import tensorflow as tf
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('bestmodel.h5')
  return model
model=load_model()
st.write("""
        # BRAIN TUMOUR DETECTION
""")  

file = st.file_uploader("Upload the MRI Scan",type=["jpg","jpeg","png"])
st.image(file)
from PIL import Image  
import PIL  
picture = Image.open(file)  
picture = picture.save("./saved.jpg")
if file is None:
  st.text("Please upload an image file")
else:
  from keras.preprocessing import image
  from keras.preprocessing.image import load_img, img_to_array
  path = "saved.jpg"
  img = load_img(path,target_size=(224,224))
  input_arr = img_to_array(img)/255

  input_arr.shape

  input_arr=np.expand_dims(input_arr, axis=0)

  pred = model.predict(input_arr)[0][0]


  if pred < 0.5:
    string="The MRI has a Tumor"
  else:
    string="The MRI is not having a Tumour"
  st.success(string)   