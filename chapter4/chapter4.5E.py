# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:12:23 2021

@author: KT
"""

'''
import tensorflow as tf # 引入tensorflow相关包
constant_a = tf.constant('Hello World!') # 定义常量
with tf.Session() as session:
  print(session.run(constant_a)) # 运行图，并获取constant_a的执行结果
'''
  
import tensorflow as tf # 引入tensorflow相关包
placeholder_a = tf.placeholder(tf.float32) # 定义placeholder实例
placeholder_b = tf.placeholder(tf.float32)
add_result = tf.add(placeholder_a, placeholder_b) # OP使两值相加
multiply_result = tf.multiply(placeholder_a, placeholder_b) # OP使两值相加
with tf.Session() as session:
  # 运行图，获取执行结果
  print(session.run(add_result, feed_dict = {placeholder_a: 1.0, placeholder_b: 2.0})) # 获取单个值
  print(session.run([add_result, multiply_result], feed_dict = {placeholder_a: 3.0, placeholder_b: 4.0})) # 获取多个值
  
  