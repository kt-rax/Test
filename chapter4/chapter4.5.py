# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:00:52 2021

@author: KT
"""
'''
import matplotlib.pyplot as plt
import cv2
import pylab
import numpy as np
img = plt.imread("mwz.jpg")
plt.imshow(img)
#pylab.show(img)
filter1 = np.array([[-1,0,1],[-2,0,2],[-1,0.1]])
filter2 = np.array([[-1,-1,-1,0,1],[-1,-1,0,1,1],[-1,0,1,1,1]])
res = cv2.filter2D(img,-1,filter2)
plt.imshow(res)
plt.imsave('result1.jpg',res)
#pylab.show()

'''

import tensorflow as tf
#import os
#import matplotlib.pylab as plt

'''
(-)数据预处理
'''
#1.读入图片
src_img = tf.gfile.FastGFile('mwz.jpg','rb').read()
#2.解码图片，得到HWC模式的RGB图像
img_rgb_uint8 = tf.image.decode_jpeg(src_img)
print(img_rgb_uint8)
#3.转换数据格式，从uint8转换为float32
img_rgb_float32 = tf.image.convert_image_dtype(img_rgb_uint8, dtype=tf.float32)
#4.转换shape，由HWC转换为CHW
img_CHW_ = tf.transpose(img_rgb_float32, perm=[2,0,1])
#5.扩展一个维度，由CHW转换为NCHW
img_NCHW_ = tf.expand_dims(img_CHW_, 0)


'''
(二)开始进行卷积处理计算
'''
#定义一个卷积核，参数2为[核宽，核高，输入通道数]；定义一个偏置，参数2为[等同于输出通道数]
tf.reset_default_graph()
w = tf.get_variable("weight",shape=[3,3,3,3],initializer=tf.truncated_normal_initializer(stddev=0.1))

#执行卷积 
conv_ = tf.nn.conv2d(img_NCHW_,w,strides=[1,1,1,1],padding='SAME',data_format='NCHW')


'''
（三）数据提取操作
'''
#降维，便于保存，由NCHW降为CHW
img_to_save_CHW_ = tf.squeeze(conv_,0)
#转换shape，由CHW转为HWC（原始图片读入就是HWC）
img_to_save_HWC_ = tf.transpose(img_to_save_CHW_, perm=[1,2,0])   
#转换图片数据格式，由uint8转为float32
enconde_image_u8 = tf.image.convert_image_dtype(img_to_save_CHW_,dtype=tf.uint8)
#编码为jpg
conv_img = tf.image.encode_jpeg(enconde_image_u8)


'''
(四 session)
'''     
#config = tf.ConfigProto()

#with tf.Session(config = config) as sess:
with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    img_CHW = sess.run(img_CHW_)
    print(img_CHW.shape)
    img_NCHW = sess.run(img_NCHW_)
    print(img_NCHW.shape)
    conv = sess.run(conv_)
    print(conv.shape)
    img_to_save = sess.run(img_to_save_CHW_)
    print('img_to_save shape:',img_to_save.shape)
    img_to_save_HWC = sess.run(img_to_save_HWC_)
    print('img_to_save_HWC shape:',img_to_save_HWC)
    #图片编码可以输入两种格式，一种是Tensor，一种是ndarray类型
    #必须是HWC模式
    save_image = sess.run(conv_img)
    #保存文件到本地文件
    with tf.gfile.GFile('mwz_4,jpg','wb') as f:
        f.write(save_image)


'''
filter_shape = [1,2,2,1]
strides = [1,2,2,1]
padding = 'VALID'
pool = tf.nn.max_pool(img_NCHW, filter_shape, strides, padding)

print(pool)
'''

















