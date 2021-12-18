# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 19:49:34 2021

@author: KT
"""
#from gensim.test.utils import common_texts,get_tmpfile
from __future__ import print_function
import matplotlib.pyplot as plt
from chainer.datasets import mnist

train,test = mnist.get_mnist(withlabel=True,ndim=1)

x,t = train[0]
plt.imshow(x.reshape(28,28), cmap='gray')
plt.savefig('5.png')
print('label:',t)

