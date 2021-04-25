# -*- coding: utf-8 -*-

import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras.utils
import PIL
import os


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
while True:
    ret, img = cap.read()
    img = img.copy()
    cv2.imwrite('try.jpg',img)
    cv2.imshow('Camera', img,)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
