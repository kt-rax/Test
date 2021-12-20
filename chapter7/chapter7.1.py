# -*- coding: utf-8 -*-
'''
#单词分割 
import nltk
#nltk.download()
sentence = "At eight o'clock on Thursday moring Arthur didn't fell very good."
tokens = nltk.word_tokenize(sentence)
print(tokens)

from textblob import TextBlob
text = "I fell happy today. I feel sad."
blob = TextBlob(text)
#第一句的情感分析
first = blob.sentences[0].sentiment
#第二的情感分析 
second = blob.sentences[1].sentiment

print(second)

#总的
all = blob.sentiment
print(all)

import jieba
from gensim import corpora
documents = ['Genism 也是一个用于关于自然语言处理的python']
def word_cuting(doc):
    seg = [jieba.lcut(w) for w in doc]
    return seg

texts = word_cuting(documents)
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)

#词切割 
import jieba
str = '我来到了北京清华大学'
seg_list = jieba.cut(str,cut_all=True)
print('Full Mode:','/'.join(seg_list))

#词的分割与标准 
import jieba
import jieba.analyse
import jieba.posseg
sentence_seged = jieba.posseg.cut('世界坐标系=> 相机坐标系，可以理解为相机放的位置跟世界坐标原点位置不同，而且相机还会有角度上的偏差（pitch，yaw，roll）')
outstr = ''
for x in sentence_seged:
    outstr += "{}/{},".format(x.word,x.flag)
print(outstr)

import jieba
import jieba.analyse

str = "继承特点子类默认拥有父类的所有属性和方法子类重写父类同名方法和属性"
#str = '我来到了北京清华大学'
tags = jieba.analyse.extract_tags(str,topK=33)
print('/'.join(tags))

#关键字抽取
import jieba
import jieba.analyse
filename = 'LICENSE'
content = open(filename,mode='r',encoding='UTF-8').read()
tags = jieba.analyse.extract_tags(content,topK=10)
print('/'.join(tags))


#基于TextRank算法的关键词抽取 
import jieba
import jieba.analyse
test = '线程是程序执行时的最小单位，他是进测的一个执行流，在多CPU环境下就允许多个\
线程同时运行，同样多线程也可以实现并发操作，每个请求分配一个线程来处理'

tags = jieba.analyse.textrank(test,withWeight=True)
for x,w in tags:
    print('%s   %s' %(x,w))


from gensim import corpora
import jieba

documents = ['物联网是新一代信息技术的重要组成部分','也是信息化时代的重要发展阶段']
def word_cut(doc):
    seg_word = [jieba.lcut(w) for w in doc]
    return seg_word
texts = word_cut(documents)
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)



#词性标准 
from textblob import TextBlob
str1 = TextBlob('python is a high-level,general-purpose programming languae.')
print(str1.tags)


#情感分析 
from textblob import TextBlob
str1 = TextBlob('I am glad')
print(str1.sentiment)
str2 = TextBlob('I am sad')
print(str2.sentiment)


from  textblob import TextBlob
zen = TextBlob("I am gald." "I am complex")
print(zen.sentences)



#中文情感分析库snowplp
from snownlp import SnowNLP
text = u"我今天很快乐。我今天很悲伤"  #使用unicode编码 
s = SnowNLP(text)
print(SnowNLP(s.sentences[0]).sentiments)
print(SnowNLP(s.sentences[1]).sentiments)

 

#文本特走提取，skearn的子库 CountVectorizer 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
texts = ["我 爱 北京","北京 是 首都","中国  北京"]
cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)
print(cv.vocabulary_)
print(cv_fit)

print(cv_fit.toarray())

#TfidfVectorizer的分本特走提取 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
texts = ["我 爱 北京","北京 是 首都","中国  北京"] 
cv = TfidfVectorizer()
cv_fit = cv.fit_transform(texts)
print(cv.vocabulary_)
print(cv_fit)
print(cv_fit.toarray())


#新闻分类 
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
#导入文本特征向量转化模块 
from sklearn.feature_extraction.text import CountVectorizer
#导入朴素贝叶斯模型 
from sklearn.naive_bayes import MultinomialNB
#模型评估模块
from sklearn.metrics import classification_report

#1.读取数据
news = fetch_20newsgroups(subset='all')
#2.分割数据
x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)
#3.贝叶斯分类器对新闻进行预测
#文本转换为特走 
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
#初始化贝叶斯模型 
mnb = MultinomialNB()
#训练集合上进行训练，估计参数
mnb.fit(x_train,y_train)
y_predict = mnb.predict(x_test)
#4.模型评估
print("准确率：",mnb.score(x_test,y_test))
#print("其他指标：\n",classification_report(y_test,y_predict,target_names=news.target_names)) 

print("其他指标：\n",classification_report(y_test, y_predict, target_names=news.target_names))



#import en_core_web_sm
#parser = en_core_web_sm.load()

#spacy的流水线和属性 
import spacy
from spacy.lang.en import English
import en_core_web_sm
parser = en_core_web_sm.load()
sentences = "There is an art,it says,or rather,a knack to flying. The knack lies in learning\
    how to throw yourself at the ground and miss. In the beginning the Universe was created.\
    This has made a lot of people very angry and been widely regarded as a bed movie"
print("解析文本中包含的句子")
sents = [sent for sent in parser(sentences).sents]
for x in sents:
    print(x)
print("-*-*"*20)
#分词
print()
tokens = [token for token in sents[0] if len(token)>1]
print(tokens)
print("-*-*"*20)
print()
#词性还原
lemma_tokens = [token.lemma_ for token in sents[0] if len(token)>1]
print(lemma_tokens)
print("-*-*"*20)
print()
#简化版的词性标注
pos_tokens = [token.pos_ for token in sents[0] if len(token)>1]
print(pos_tokens)
print("-*-*"*20)
#词性标注的细节版
tag_tokens = [token.tag_ for token in sents[0] if len(token)>1]
print(tag_tokens)
print("-*-*"*20)
#依存分析
dep_tokens = [token.dep_ for token in sents[0] if len(token)>1]
print(dep_tokens)
print("-*-*"*20)
print("名词块分析")
doc = parser(u"Autonomous cars shift insurance liability toward manufacturers")
#获取名词块文本
chunk_text = [chunk.text for chunk in doc.noun_chunks]
print(chunk_text)
print("-*-*"*20)
#获取名词块根节点的文本
chunk_root_text = [chunk.root for chunk in doc.noun_chunks]
print(chunk_root_text)  
print("-*-*"*20)
#依存分析
chunk_root_dep_ = [chunk.root.dep_ for chunk in doc.noun_chunks]
print(chunk_root_dep_)   
print("-*-*"*20)
#
chunk_root_head_text = [chunk.root.head.text for chunk in doc.noun_chunks]
print(chunk_root_head_text)
print("-*-*"*20)       


##
##TypeError: dump() missing 1 required positional argument: 'fp'
from __future__ import print_function,unicode_literals
import json
import requests

CLASSIFY_URL = 'https://api.bosonnlp.com/classify/analysis'
s =['俄否决安理会谴责叙利亚战机空袭平民','邓紫棋谈男朋友：我觉得我比他唱的号','facebook收购印度初创公司']

data = json.dumps(s)
headers = {
    'X-Token':'YOUR_API_TOKEN',
    'Content-Type':'application/json'
    }
resp = requests.post(CLASSIFY_URL,headers = headers,data=data.encode('utf-8'))
print(resp.text)
'''

from __future__ import print_function,unicode_literals
import json
import requests
HEADERS = {'X-Token':'YOUR_API_TOKEN','Content-Type':'application/json'}

SENTIMENT_URL = 'http://api.bosonnlp.com/sentiment/analysis'
def main():
    print('读入数据...')
    with open('text_sentiment.txt','rb') as f:
        docs = [line.decode('utf-8') for line in f if line]
    print('正在上传数据...%s'%(len(docs)))
    for i in range(0,len(docs),100):
        data = docs[i:i+100]
        all_proba = requests.post(SENTIMENT_URL,headers=HEADERS,data=json.dumps(data).encode('utf-8')).json()
        text_with_proba = zip(data,all_proba)
        sort_text = sorted(text_with_proba,key=lambda x: x[1][1],reverse=True)
        for text,sentiment in sort_text:
            print(sentiment,text)

if __name__ == '__main__':
    main()
    








































