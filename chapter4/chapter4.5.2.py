# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:11:00 2021

@author: KT
"""
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout

def LeNet():
    model1 = Sequential()
    model1.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    model1.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model1.add(MaxPooling2D(pool_size=(2,2)))
    model1.add(Flatten())   
    model1.add(Dense(100,activatiion='relu'))
    model1.add(Dense(100,activatiion='softmax'))
    return model1
    
def AlexNet():
    model1 = Sequential()
    model1.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model1.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='uniform'))
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model1.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model1.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model1.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model1.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model1.add(Dense(4096,activation='relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(4096,acitvation='relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(1000,activation='softmax'))
    return model1