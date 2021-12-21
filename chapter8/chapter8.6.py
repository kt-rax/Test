# -*- coding: utf-8 -*-

import cv2
import dlib
#使用特征提取器get_frontal_face_detector()
detector = dlib.get_frontal_face_detector()

#dilb的点模型，特征提取器
predictor = dlib.shape_predictor('shape_predictor_face_landmakr.dat')

#加载图片
img = io.imread('r1.jpg')

#生成dlib的图像窗口
win = dlib.image_window()
win.clear_overlay()

#特征提取器的实例化
dets = detector(img,1)
print('人脸数'，len(dets))
for k,d in enumerate(dets):
    print('第'，k+1,'个人脸d的坐标'，'left:',d.left(),
          'right:'，d.right(),'top:'，d.top(),'bottom:',d.bottom())
    width = d.right() - d.lefe()
    height = d.bottom() - d.top()
    print('人脸面积为：',(width*height))