# -*- coding: utf-8 -*-
'''
import cv2
#1.定义人脸检测的分类器  
face_cascade = cv2.CascadeClassifier(r'D:\program\anaconda3\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml')
#2.读取图片 
img = cv2.imread('xx.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#3.探测图片中的人脸  
faces = face_cascade.detectMultiScale(gray)
print('发现{0}个人眼！'.format(len(faces)))
for (x,y,w,h) in faces:
    cv2.circle(img,(int((x+x+w)/2),int((y+y+h)/2)),int(w/2),(0,255,0),2)
    cv2.namedWindow('人眼识别', 0)
    cv2.resizeWindow('人眼识别', 800, 1200)
    cv2.imshow('人眼识别',img)
    cv2.imwrite('Qxx.jpg',img)
    cv2.waitKey(0)
'''

'''
图片相似度检测 
'''
import numpy as np
from matplotlib import pyplot as plt
import cv2
#1.定义灰度值计算函数 
def classify_gray_hist(image1,image2,size=(256,256)):
    #2.重定义图像的大小 
    image1 = cv2.resize(image1,size)
    image2 = cv2.resize(image2,size)
    #3.获取每个像素点的频数值 
    hist1 = cv2.calcHist([image1],[0], None, [256], [0.0,255.0])   
    hist2 = cv2.calcHist([image2],[0],None, [256], [0.0,255.0])
    #4.画直方图 
    plt.plot(range(256),hist1,'r')
    plt.plot(range(256),hist2,'b')
    plt.show()
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 -abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
        else:
            degree = degree + 1
    degree = degree/len(hist1)
    return degree
#6.定义计算单通道直方图相似度的函数
def calculate(image1,image2):
    hist1 = cv2.calcHist([image1], [0] ,None, [256], [0.0,255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0,255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i]  != hist2[i]:
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i]))
        else:
            degree = degree + 1
    degree = degree/len(hist1)
    return degree
#7.计算直方图的重合度 
def classify_hist_with_split(image1,image2,size=(256,256)):
    image1 = cv2.resize(image1,size)
    image2 = cv2.resize(image2,size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0 
    for im1,im2 in zip(sub_image1,sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data/3
    return sub_data
#8.定义平均哈希算法计算函数
def classify_aHash(image1,image2):
    image1 = cv2.resize(image1,(8,8))
    image2 = cv2.resize(image2,(8,8))
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1,hash2)
#9.定义函数
def classify_pHash(image1,image2):
    image1 = cv2.resize(image1,(32,32))
    image2 = cv2.resize(image2,(32,32))
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    dct1_roi = dct1[0:8,0:8]
    dct2_roi = dct2[0:8,0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1,hash2)
#10.输入灰度图 
def getHash(image):
    average = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > average:
                hash.append(1)
            else:
                hash.append(0)
    return hash
#11.计算汉明距离 
def Hamming_distance(hash1,hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

if __name__ == '__main__':
    img1 = cv2.imread('A.jpg')
    cv2.namedWindow('A', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('A', 800, 1200)
    cv2.imshow('A',img1)
    img2 = cv2.imread('B.jpg')
    cv2.namedWindow('B', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('B', 800, 1200)
    cv2.imshow('B',img2)
    degree1 = classify_gray_hist(img1,img2)
    degree2 = classify_hist_with_split(img1,img2)
    degree3 = classify_aHash(img1,img2)
    degree4 = classify_pHash(img1,img2)
    print('gray hist is :{}\n split hist is {}\n aHash is :{}\n pHash is : {}\n' .format(degree1,degree2,degree3,degree4))
    
    print('gray hist is :%f\n split hist is %f\n aHash is :%d\n pHash is : %d\n' %(degree1,degree2,degree3,degree4))
    
    print(f'gray hist is :{degree1}\n split hist is {degree2}\n aHash is :{degree3}\n pHash is : {degree4}\n')
    
    
    cv2.waitKey(0)































