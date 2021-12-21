# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.optimizers import SGD,RMSprop,Adagrad
from keras.utils import np_utils,plot_model
from keras.models import  Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
import tensorflow as tf
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import load_model
import h5py

#读取训练预料 
neg = pd.read_excel('neg.xls',header=None,index=None)
pos = pd.read_excel('pos.xls',header=None,index=None)
#给训练预料贴上标签 
pos['mark'] = 1
neg['mark'] = 0
#合并预料
pn = pd.concat([pos,neg],ignore_index=True)

#计算预料数目
neglen = len(neg)
poslen = len(pos)

#定义分词函数 
cw = lambda x: list(jieba.cut(x))
pn['words'] = pn[0].apply(cw)

#读取评论内容
comment = pd.read_excel('sum.xls')
#仅读取非空评论 
comment = comment[comment['rateContent'].notnull()]
#评论分词
comment['words'] = comment['rateContent'].apply(cw)

d2v_train = pd.concat([pn['words'],comment['words']],ignore_index=True)

#将所有的词语整合在一起
w = []

for i in d2v_train:
    w.extend(i)

#统计词的出现次数 
dict = pd.DataFrame(pd.Series(w).value_counts())
del w,d2v_train
dict['id'] = list(range(1,len(dict)+1))
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)
maxlen = 50
print('pad sequence(samples x time)')
#print('Pad sequence (samples x time'))
pn['sent'] = list(sequence.pad_sequences(pn['sent'],maxlen = maxlen))
#训练集
x = np.array(list(pn['sent']))[::2]
y = np.array(list(pn['mark']))[::2]
#测试集
xt = np.array(list(pn['sent']))[1::2]
yt = np.array(list(pn['mark']))[1::2]
#全集
xa = np.array(list(pn['sent']))
ya = np.array(list(pn['mark']))

print('Build model ...')
# 创建模型 
model = Sequential()
model.add(Embedding(len(dict)+1,256,name='layer_1_embdeding'))
model.add(LSTM(128,name='layer_2_LSTM'))
model.add(Dropout(0.5,name='layer_3_Dropout'))
model.add(Dense(1,name='layer_4_Dense'))
model.add(Activation('sigmoid',name='lyaer_5_Activation'))

# 测试模型保存与装载-1:在模型定义完但还没有训练进行保存 
#model.save('Use_model_save_model1.h5')
#model_1 = load_model('Use_model_save_model1.h5')
#model_2_1  = h5py.File('Use_model_save_model1.h5','r')  

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

# 设置训练保存断点 
log_dir="logs"
'''
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{acc:.2f}-{loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir)
'''
    # Fix AttributeError: 'Sequential' object has no attribute '_ckpt_saved_epoch'
my_First_callback =[
    EarlyStopping(patience=2),
    ModelCheckpoint(filepath='model.{epoch:02d}-{acc:.2f}-{loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]

# 模型训练并返回记录
history = model.fit(x,y,batch_size=16,nb_epoch=3,callbacks=my_First_callback)
#history = model.fit(x,y,batch_size=16,nb_epoch=2,callbacks=my_First_callback)

# 测试模型保存与装载-2,在模型训练完成后进行保存，对比位置1与位置2所保存的模型可有差别 
#model.save('Use_model_save_model2.h5')
#model_2 = load_model('Use_model_save_model2.h5')
#model_2_2  = h5py.File('Use_model_save_model2.h5','r')  

# plot模型网络结构图
plot_model(model,to_file='chapter8.4_model.png',show_shapes=True)
classes = model.predict_classes(xt)

# plot训练的精度与误差 
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.xlabel('epoch')
plt.ylabel('loss & acc')
plt.legend(['loss','acc'], loc='upper left')
plt.savefig('chapter8_4_result')
plt.imshow()

'''
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['loss'])
plt.title('accuracy & loss')
plt.xlabel('Test Step')
plt.ylabel('loss accuracy')
plt.legend(['binary_accuracy','loss'],loc='upper left')
plt.savefig('test_binary_accuracy')
plt.show()
'''
'''
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir="logs/fit/" 
#+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])
'''
























