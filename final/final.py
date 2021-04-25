# Ignore warnings
import warnings
import random
warnings.filterwarnings("ignore")

import serial
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras.utils
import PIL
from PIL import Image
from keras import backend as K
from keras.preprocessing import image
from keras.models import Sequential,Model,load_model  #用來啟動 NN
from keras.models import load_model
##from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D,Dropout,Activation,GlobalAveragePooling2D # Fully Connected Networks
import os
from numba import cuda
import tensorflow as tf

# 自動增長 GPU 記憶體用量
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 Session
tf.keras.backend.set_session(sess)

model = load_model('model_001.h5')

#arduino設定
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
COM_PORT = 'COM5'  # 請自行修改序列埠名稱
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)
catch_tku = 0
non_catch = 0

#圖片預測
def image_predict(img):
    img = image.load_img(img, target_size=(224,224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    ans = model.predict(x)[0]
    return(ans)

def cap_0():
    ret, img = cap.read()
    cap_img = img.copy()
    return (cap_img)

#arduino溝通
def py2inoON():
    ser.write(b'on\n') 
    print("啟動")
    time.sleep(0.1)
def py2inoOFF():
    ser.write(b'off\n')
    time.sleep(0.1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

while(1):
    ret, img = cap.read()
    cv2.imwrite('img.jpg',img)
    score = image_predict('img.jpg')
    score = np.around(score, 7)
    print(score)
    cv2.imshow('Camera',img)
    if score[1] >= 0.75:
        catch_tku = catch_tku+1
        if catch_tku >= 3:
            py2inoON()
            catch_tku = 0
            non_catch = 0
    else :
        non_catch = non_catch+1
        if non_catch>=2:
            non_catch = 0
            catch_tku = 0
    #time.sleep(0.2)
    py2inoOFF()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cap.release()
cv2.destroyAllWindows()

