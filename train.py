from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras.utils
import PIL
from PIL import Image
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential,Model,save_model  #用來啟動 NN
from keras.models import load_model
from keras.layers import Dense,Flatten,MaxPooling2D,Conv2D,Dropout,Activation,GlobalAveragePooling2D,Dropout# Fully Connected Networks
import os
from keras.applications import imagenet_utils
from keras.applications.resnet50 import  preprocess_input
from numba import cuda
#%%
# 資料路徑
DATASET_PATH  = './data'

# 影像大小
IMAGE_SIZE = (224,224)

# 影像類別數
NUM_CLASSES = 3

# 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
BATCH_SIZE = 32

# 凍結網路層數
FREEZE_LAYERS = 2

# Epoch 數
NUM_EPOCHS = 20

# 模型輸出儲存的檔案
WEIGHTS_FINAL = 'model-vgg16-final.h5'
#%%
# 透過 data augmentation 產生訓練與驗證用的影像資料
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/val',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# 以訓練好的 ResNet50 為基礎來建立模型，
# 捨棄 ResNet50 頂層的 fully connected layers
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# 增加 DropOut layer
x = Dropout(0.5)(x)

# 增加 Dense layer，以 softmax 產生個類別的機率值
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# 設定凍結與要進行訓練的網路層
model = Model(inputs=net.input, outputs=output_layer)
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
model.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# 輸出整個網路結構
print(model.summary())
#%%
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# 設定 Keras 使用的 Session
tf.keras.backend.set_session(sess)
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
    
# 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
model.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
#callback = EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode="auto")
callback = ModelCheckpoint(('model_{epoch:03d}.h5'),verbose=1, save_weights_only=False, period=1,save_best_only=False)
# 輸出整個網路結構
#print(net_final.summary())

# 訓練模型
his=model.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches, 
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = 10,callbacks=[callback])
#%%
model.save('final_model.h5')