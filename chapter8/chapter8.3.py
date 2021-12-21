# -*- coding: utf-8 -*-
'''
from bosonnlp import BosonNLP
nlp = BosonNLP('YOUR_API_TOKEN')
nlp.sentiment('这家味道还不错')

from bosonnlp import BosonNLP
import os
nlp = BosonNLP(os.environ['BOSON_API_TOKEN'])
nlp.sentiment('这家味道还不错',mode='food')
'''
import pickle
#写入一个文件，用二进制的形式
f = open('data.pkl','wb')
#等待写入的数据
datas = {'name':'li','age':30,'high':170}
#dump函数将obj数据datas导入到file中
data_one = pickle.dump(datas,f,-1)
#f文件结束操作句柄
f.close()
#

f2 = open('data.pkl','rb')
print(pickle.load(f2))
f2.close()