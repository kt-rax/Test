# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 17:29:48 2021

@author: KT
"""

'''
import cv2

#获取内置检测器
face_cascade = cv2.CascadeClassifier(r'D:\program\anaconda3\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml')
#读取图片D:\program\anaconda3\Graphviz\bin
img = cv2.imread('mwz.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
#探测图片中的人脸
eyes = face_cascade.detectMultiScale(gray)
print('发现{format(len(eyes)}个眼睛')
for (x,y,w,h) in eyes:
    cv2.circle(img,(int((x+x+w)/2),int((y+y+h)/2)),int(w/2),(0,255,0),2)

               
cv2.imshow('Findface',img)
cv2.imshow("Find Faces" ,img)
cv2.imwrite('Q.jpg',img)
cv2.waitKey(0)
'''
import cv2
#获取内置检测器
smilePath = (r'D:\program\anaconda3\Lib\site-packages\cv2\data\haarcascade_smile.xml')
smileCascade = cv2.CascadeClassifier(smilePath)
face_cascade = cv2.CascadeClassifier(r'D:\program\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')

img = cv2.imread('mwz.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255), 2)
    roi_gray = gray[y:y+h,x:x+w] 
    roi_color = img[y:y+h,x:x+w]
    smile = smileCascade.detectMultiScale(roi_gray,scaleFactor=1.16,minNeighbors=60,minSize=(25,25),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x2,y2,w2,h2) in smile:
        cv2.rectangle(roi_color, (x2,y2), (x2+w2,y2+h2),(255,0,0),2)
        cv2.putText(img,'smile',(x,y-7),3,1.2,(0,255,0),2,cv2.LINE_AA)                   
        
cv2.imshow('smile test',img)
cv2.imwrite('smile.jpg', img)
c = cv2.waitKey(0)

