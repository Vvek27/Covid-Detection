import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.elements.image import image_to_url
from streamlit_lottie import st_lottie
import requests

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import os,random
import warnings
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img

from tensorflow.python.ops.array_ops import shape_v2
warnings.filterwarnings("ignore")

st.set_page_config(initial_sidebar_state="collapsed")

html_temp = """
    <div style="background-color:grey;padding:8px">
    <h2 style="color:white;text-align:center;">Covid Detection</h2>
    </div>"""
st.markdown(html_temp,unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://image.freepik.com/free-photo/gray-abstract-wireframe-technology-background_53876-101941.jpg")
 }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # sidebar
    st.set_option('deprecation.showPyplotGlobalUse', False)
    activities=["Select activity","Images","Filters","Prediction","About"]
    choice=st.sidebar.radio("",activities)
    if choice=="Select activity":
        activity()
    if choice=="Prediction":
        Prediction()


def activity():
    def lottie_file(url:str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_hello=lottie_file("https://assets6.lottiefiles.com/packages/lf20_kltum0us.json")
    st_lottie( lottie_hello, speed=1, reverse=False,loop=True,quality="low",
    renderer="svg")

def Prediction():
    def classify(image,model):
        #load model
        my_model=load_model(model)
        prediction=my_model.predict(img)
        return prediction
    uploaded_file = st.file_uploader("Choose a X-ray image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        if st.checkbox("File information"):
            d={"File name":[uploaded_file.name],"File size":[uploaded_file.size],
            "File type":[uploaded_file.type]}
            df=pd.DataFrame(d)
            st.write(df)
        # save the particular file
        with open (uploaded_file.name,"wb") as f:
            f.write(uploaded_file.getbuffer())
        image_name=uploaded_file.name
        img_path="Covid-Detection" + "/" + str(image_name)
        img=load_img(img_path,target_size=(256,256))
        if st.checkbox("Show image"):
            st.image(img,width=400)
        img=image.img_to_array(img)/255
        img=np.array([img])# for getting 1 count for image
        result=classify(img,"covid_model.h5")
        if st.button("Classify"):
            if result[0]<0.5:
                st.warning("You have covid")
                st.write(result)
            else:
                st.success("You are healthy")
                st.write(result)
       
if __name__=="__main__":
    main()
