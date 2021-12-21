# -*- coding: utf-8 -*-
#keras 构建模型方法1 sequential模型 ：层间线性顺序关系 
''' 
from keras.models import Sequential
from keras.layers import Dense,Activation
msodel = Sequential()
model.add(Dense(64,input_shape = (784,)))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('softmax'))
#keras 构建模型方法2 model ：灵活 ，可以设计复杂任意拓扑的神经网络
from keras.layers import Input,Dense
from keras.models import Model
#定义输入层，确定输入维度
input = input(shape = (1024,))
#2个隐含层，每个都有128个神经元，使用ReLU激活函数，且由上一层作为参数
x = Dense(128,activation='relu')(input)
x = Dense(128,activation='relu')(x)
#输出层
y = Dense(16,activation='softmax')(x)
#定义模型，指定输入输出
model = Model(input = input,output = y)
#编译模型，指定优化器，损失函数，度量
model.compile(optimizer='rmsprop', loss='catagorical_crossentropy',metrics=['accuracy'])
#模型拟合，训练
model.fit(data,labels)
'''
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras import losses
model = Sequential()
#输入层
model.add(Dense(10,input_shape=(4,)))
model.add(Activation('sigmoid'))
#隐藏
model.add(Dense(13))
model.add(Activation('relu'))
model.add(Activation('tanh'))
#输出层
model.add(Dense(9))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss=losses.categorical_crossentropy,metrics=['accuracy'])
model.summary()

































