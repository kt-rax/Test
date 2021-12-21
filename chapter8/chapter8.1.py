# -*- coding: utf-8 -*-
'''
#词集模型 
import jieba
import  jieba.analyse
import jieba.posseg
def dosegment_all(sentence):
    sentence_seged = jieba.posseg.cut(sentence.strip())
    outstr = ''
    for x in sentence_seged:
        outstr += x.word+' '
    return outstr
print(dosegment_all('动态库： 是一个目标文件，包含代码和数据，它可以在程序运行时动态的加载并链接。修改动态库不需要重新编译目标文件，只需要更新动态库即可。动态库还可以同时被多个进程使用。在linux下生成动态库 gcc -c a.c  -fPIC -o a.o     gcc -shared -fPIC a.o -o a.so.     这里的PIC含义就是生成位置无关代码，动态库允许动态装入修改，这就必须要保证动态库的代码被装入时，可执行程序不依赖与动态库被装入的位置，即使动态库的长度发生变化也不会影响调用它的程序。'))

#词袋模型
from sklearn.feature_extraction.text import  CountVectorizer
corpus = ['动态库： 是一个目标文件，包含代码和数据，它可以在程序运行时动态的加载并链接。','修改动态库不需要重新编译目标文件，只需要更新动态库即可。动态库还可以同时被多个进程使用。','在linux下生成动态库 gcc -c a.c  -fPIC -o a.o     gcc -shared -fPIC a.o -o a.so.     这里的PIC含义就是生成位置无关代码，动态库允许动态装入修改，这就必须要保证动态库的代码被装入时，可执行程序不依赖与动态库被装入的位置，即使动态库的长度发生变化也不会影响调用它的程序。']
vec = CountVectorizer(min_df=1)
X = vec.fit_transform(corpus)
print(X)
fnames = vec.get_feature_names()
print(fnames)
arr = X.toarray()
print(arr)

#TF-IDF模型
from sklearn.feature_extraction.text import TfidfTransformer
from  sklearn.feature_extraction.text import CountVectorizer

corpus = ['动态库： 是一个目标文件，包含代码和数据，它可以在程序运行时动态的加载并链接。','修改动态库不需要重新编译目标文件，只需要更新动态库即可。动态库还可以同时被多个进程使用。','在linux下生成动态库 gcc -c a.c  -fPIC -o a.o     gcc -shared -fPIC a.o -o a.so.     这里的PIC含义就是生成位置无关代码，动态库允许动态装入修改，这就必须要保证动态库的代码被装入时，可执行程序不依赖与动态库被装入的位置，即使动态库的长度发生变化也不会影响调用它的程序。']
#词袋化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
#TF-IDF
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
print(tfidf)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
corpus = ['动态库： 是一个目标文件，包含代码和数据，它可以在程序运行时动态的加载并链接。','修改动态库不需要重新编译目标文件，只需要更新动态库即可。动态库还可以同时被多个进程使用。','在linux下生成动态库 gcc -c a.c  -fPIC -o a.o     gcc -shared -fPIC a.o -o a.so.     这里的PIC含义就是生成位置无关代码，动态库允许动态装入修改，这就必须要保证动态库的代码被装入时，可执行程序不依赖与动态库被装入的位置，即使动态库的长度发生变化也不会影响调用它的程序。']
result = tfidf.fit_transform(corpus)
print(result)

#将单词转换为特征向量 
import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer
count = CountVectorizer(ngram_range=((1,1))
#text = ['I come to Beijing']
#bag = count.fit_transform(text)
print(count.vocabulary_)
print(bag.toarray())
   
from sklearn.feature_extraction.text import CountVectorizer               
import  numpy as np
count = CountVectorizer(ngram_range=(1,1))
text1 = np.array(['I come to Beijing','beijing is captial','I like beijing'])
print(np.array(['I come to Beijing','beijing is captial','I like beijing']))
bag = count.fit_transform(text1)
print(count.vocabulary_)
print(bag.toarray())

from sklearn.feature_extraction.text import CountVectorizer
texts = ['dog cat fish','fox pig tiger','fish bird','bird']
cvt = CountVectorizer()
cvt_fit = cvt.fit_transform(texts)

print(cvt.get_feature_names())
print(cvt.vocabulary_)
print(cvt_fit)
print(cvt_fit.toarray())
print(cvt_fit.toarray().sum(axis=0))

import cv2
import numpy as np
img = cv2.imread('B.jpg')
cv2.imshow('src',img)
print(img.shape)
print(img.size)
print(img.dtype)
print(img)


img = img.astype('float')/255.0
print(img.dtype)
print(img)

from PIL import  Image
img3 = Image.open('B.jpg')
arr = np.array(img3)
print(arr)
cv2.waitKey()


#TF-IDF计算单词关联度 
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(ngram_range=(1,1))
text = np.array(['I come to Beijing','Beijing s captial','I like Beijing'])
bag = count.fit_transform(text)
tfidf = TfidfTransformer()
np.set_printoptions(2)
print(tfidf.fit_transform(bag).toarray())
'''
import jieba
import math
import numpy as np
filename = 'title.txt' #语料库
filename2 = 'stopwords.txt'  #停用词表
def stopwordlist():
    stopwords = [line.strip() for line in open(filename2,encoding='UTF-8').readlines()]
    return stopwords

stop_list = stopwordlist()
def get_dic_input(str):
    dic = {}
    cut = jieba.cut(str)
    list_word = (','.join(cut)).split(',')
    for key in list_word:
        if key in stop_list:
            list_word.remove(key)
    length_input = len(list_word)
    for key in list_word:
        dic[key] = 0
    return dic,length_input

def get_tf_id(filename):
    s = input('请输入要检索的关键词句：')
    dic_input_idf,length_input = get_dic_input(s)
    f = open(filename,'r',encoding='utf-8')
    list_tf = []
    list_idf = []
    word_vector1 = np.zeros(length_input)
    word_vector2 = np.zeros(length_input)
    lines = f.readlines()
    length_essay = len(lines)
    f.close()
    for key in dic_input_idf:
    #计算出每个词的id值依次存储在list_idf中 
        for line in lines:
            if key in line.split():
                dic_input_idf[key] += 1
        list_idf.append(math.log(length_essay/(dic_input_idf[key]+1)))
    for i in range(length_input):
    #将idf值存储在矩阵向量中 
        word_vector1[i] = list_idf.pop()
    #依次计算每个词在每行的tf值依次存储在list_tf中 
    for line in lines:
        length = len(line.split())
        dic_input_tf,length_input = get_dic_input(s)
        for key in line.split():
            #去掉文章中的停用词 
            if key in stop_list:
                length -= 1
            if key in dic_input_tf:
                dic_input_tf[key] += 1
        for key in dic_input_tf:
            tf = dic_input_tf[key]/length
            list_tf.append(tf)
        #将每行tf值存储在矩阵向量中 
        for i in range(length_input):
            word_vector2[i] = list_tf.pop()
        tf_idf = float(np.sum(word_vector2*word_vector1))
        if tf_idf>0.3:
        #筛选出相似度高的文章 
             print('tf_idf值：',tf_idf)
             print('文章：',line)

get_tf_id(filename)
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
















