# -*- coding: utf-8 -*-
from snownlp import SnowNLP

#处理文本
text = '北京故宫是中国明清两代的皇家宫殿，旧称为紫禁城，位于北京中轴线的中心，是中国古代宫廷建筑之精华'

s = SnowNLP(text)
print(s.words)
print(s.sentences)

ss = SnowNLP(u'北京故宫是中国明清两代的皇家宫殿，旧称为紫禁城，位于北京中轴线的中心，是中国古代宫廷建筑之精华')
print(ss.words)
print(ss.sentences)