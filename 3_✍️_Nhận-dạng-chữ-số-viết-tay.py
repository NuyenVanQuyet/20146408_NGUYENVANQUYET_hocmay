import tkinter as tk
from PIL import ImageTk, Image
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2

import base64
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#img = get_img_as_base64("image.jpg")
st.set_page_config(page_title="Nhận dạng chữ số viết tay")
#page_bg_img = f"""
#<style>
#[data-testid="stSidebar"] > div:first-child {{
#background-image: url("data:image/png;base64,{img}");
#background-position: 50% 45%;
#background-size: 400%;
#}}
#/style>
#"""
#st.markdown(page_bg_img, unsafe_allow_html=True
#)

st.title("Nhận dạng chữ số viết tay")
st.sidebar.header("Nhận dạng chữ số viết tay")
st.balloons()

model_architecture = "digit_config.json"
model_weights = "digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights) 

optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"]) 

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test_image = X_test

RESHAPED = 784

X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')

#normalize in [0,1]
X_test /= 255

def main():
    col1, col2=st.columns(2)
    with col1:
        get_pic=st.button("Tạo ảnh")
    if get_pic:
        image = tao_anh()
        st.image(image, width=500, caption='Ảnh chữ số viết tay')
    with col2:
        get_dect=st.button("Nhận dạng")
    if get_dect:
        st.image('digit_random.jpg', width=500, caption='Kết quả nhận dạng chữ số')
        image = Image.open('digit_random.jpg')
        prediction = nhan_dang(image)
        results = np.argmax(prediction, axis=1)
        columns = st.columns(15)
        for i in range(len(results)):
            with columns[i%15]:
                st.write(results[i], font_size=50, bg_color='green', text_color='black', style='text-align: center')

def tao_anh():
    index = np.random.randint(0, 9999, 150)
    digit_random = np.zeros((10*28, 15*28), dtype = np.uint8)
    for i in range(0, 150):
        m = i // 15
        n = i % 15
        digit_random[m*28:(m+1)*28, n*28:(n+1)*28] = X_test_image[index[i]] 
    cv2.imwrite('digit_random.jpg', digit_random)
    image = Image.open('digit_random.jpg') 
    return image

def nhan_dang(image):
    X_test_sample = np.zeros((150,784), dtype = np.float32)
    for i in range(0, 150):
        row = i // 15
        col = i % 15
        x = col * 28
        y = row * 28
        digit = image.crop((x, y, x+28, y+28))
        X_test_sample[i] = np.asarray(digit).reshape(1, -1)
    prediction = model.predict(X_test_sample)
    return prediction

if __name__ == "__main__":
    main()
