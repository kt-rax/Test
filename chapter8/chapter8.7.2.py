# -*- coding: utf-8 -*-
# Base Version: chapter8.7.py
# change point: define the keras model by hand
# change time : Oct-05-2021
# change owner: KT
import cv2
import sys
import json
import time
import numpy as np
from keras.models import model_from_json
from keras.utils import plot_model
from keras.models import  Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import Adam
#from keras.layers.embeddings import embedding
from keras.models import load_model
import tensorflow as tf

input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=input_shape[1:])(x)

#第一步 ：defeine the model -- OK
model2 = Sequential()
model2.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',input_shape=(48,48,1),name='Third_convolution2d_10'))
model2.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='convolution2d_11'))
model2.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='convolution2d_12'))
model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='maxpooling2d_4'))

model2.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='convolution2d_13'))
model2.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='convolution2d_14'))
model2.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='convolution2d_15'))
model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='maxpooling2d_5'))

model2.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='convolution2d_16'))
model2.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='convolution2d_17'))
model2.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='convolution2d_18'))
model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid',name='maxpooling2d_6'))

model2.add(Flatten(name='fatten_2'))

model2.add(Dense(256,activation='relu',name='dense_4'))
model2.add(Dropout(0.3,name='Dropout_3'))

model2.add(Dense(256,name='dense_5'))
model2.add(Dropout(0.3,name='dropout_4'))

model2.add(Dense(6,activation='softmax',name='dense_6'))

plot_model(model2,to_file='chapter8.7.2.model_hand4.png',show_shapes=True)

#第二步：对模型进行训练 
'''
model2.compile(optimizer=Adam(learning_rate=0.0010000000474974513,beta_1=0.8999999761581421,
                              beta_2= 0.9990000128746033,epsilon=11e-08,name='Adam'), loss='categorice_crossentropy',metrics=['accuracy'])
model2.fit(data,lablels)

model2.fit(x=None,y=None,atch_size=None,epochs=1,verbose="auto",callbacks=None,validation_split=0.0,validation_data=None,shuffle=True, 
    class_weight=None,sample_weight=None,initial_epoch=0,steps_per_epoch=None,validation_steps=None,validation_batch_size=None,
    validation_freq=1,max_queue_size=10,workers=1,use_multiprocessing=False,)
'''


'''
# Camera Func test --- OK
cap = cv2.VideoCapture(0)
    
while True: 
    
    ret,img=cap.read()
    
    cv2.imshow('Video', img)
    
    if(cv2.waitKey(10) & 0xFF == ord('b')):
        break
'''   
emotion_lables = ['angry','fear','happy','sad','surprise','neutral']
cascPath = sys.argv[0]
faceCascade = cv2.CascadeClassifier(r'D:\program\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalcatface.xml')

# load json and create model arch
json_file = open('model.json','r')
load_model_json = json_file.read()
json_file.close()
model = model_from_json(load_model_json)
plot_model(model,to_file='chapter8.7.png',show_shapes=True)

# load weights into new model
model.load_weights('model.h5')

def predict_emotion(face_image_gray):
    # a single cropped face
    resized_img = cv2.resize(face_image_gray, (48,48),interpolation = cv2.INTER_AREA)
    # cv2.imwrite(str(index)+'.png',resized_img)
    image = resized_img.reshape(1,1,48,48)
    list_of_list = model.predict(image,batch_size=1,verbose=1)
    angry,fear,happy,sad,surprise,neutral = [prob for lst in list_of_list for prob in lst]
    return [angry,fear,happy,sad,surprise,neutral]

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret,frame = video_capture.read()
    img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY,1)
    faces = faceCascade.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors=5,minSize=(10,10))
    emotions = []

    # Draw a rectangle around the faces
    for(x,y,w,h) in faces:
        face_image_gray = img_gray[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,9),2)
        angry,fear,happy,sad,surprise,neutral = predict_emotion(face_image_gray)
        with open('emotion.txt','a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format(time.time(),angry,fear,happy,sad,surprise,neutral))
        print('angry, fear, happy, sad, surprise, neutral=',angry, fear, happy, sad, surprise, neutral)    
    # Disply the resulting frame
    cv2.imshow('Video',frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
        # when everything is done,release the capture
video_capture.release()
cv2.destroyAllWindows()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            