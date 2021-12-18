# -*- coding: utf-8 -*-
import urllib 
#import urllib2 
import requests
print('downloading with urllib')
url = 'https://github.com/tensorflow/tensorflow/blob/c565660e008c582668cb0937ca86e71fb/tensorflow/examples/image_retrain/retrain.py'
print("downloading with urllib")
urllib.request.urlretrieve(url, "retrian.py")
