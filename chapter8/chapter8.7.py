# -*- coding: utf-8 -*-
import cv2
import sys
import json
import time
import numpy as np
from keras.models import model_from_json
from keras.utils import plot_model



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
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            