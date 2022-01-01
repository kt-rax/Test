# -*- coding: utf-8 -*-

import cv2 
import numpy as np
#img = cv2.imread('che.jpg',1)
img = cv2.imread('che.jpg')
lower=[100,254,254]
upper=[124,254,254]
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower = np.array(lower,dtype='uint8')
upper = np.array(upper,dtype='uint8')

mask = cv2.cvtColor(hsv,lowerb=lower,upperb=upper)
output = cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('image',img)
cv2.imshow('image-location',output)
cv2.waitKey(0)

'''

import cv2 
import numpy as py
import matplotlib.pyplot as plt

img = cv2.imread('chepai.jpg')
cv2.imshow('orignal',img)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retv,thresh = cv2.threshold(img_gray,125,255,1)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_original,contours,-1,(0,0,250),3,lineType=cv2.LINE_AA)
cv2.imshow('Contours',img_original)
cv2.waitKey(0)
cv2.destroyAllWindow()
'''
