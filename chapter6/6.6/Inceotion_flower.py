# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:18:32 2020

@author: hefug
"""

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class Inception_v3(object):
    def __init__(self,):
    	#配置文件
        '''
        label_path = 'config/imagenet_2012_challenge_label_map_proto.pbtxt' #标签文件
        id_path = 'config/imagenet_synset_to_human_label_map.txt'#编号id
        model_path = 'config/classify_image_graph_def.pb'#模型文件
        '''
        label_path = 'imagenet_2012_challenge_label_map_proto.pbtxt'
        id_path = 'imagenet_synset_to_human_label_map.txt'
        model_path = 'classify_image_graph_def.pb'  
        
        self.__map = self.load_map(label_path, id_path)
        self.load_model(model_path)
	#魔法方法call  可直接用类名家括号的方式调用该方法
    def __call__(self, images_path='images/'):
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            for root, dirs, files in os.walk(images_path):
                if files is None:
                    print("directory is null!")
                for file in files:
                    image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
                    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式
                    predictions = np.squeeze(predictions)

                    # 打印图片路径及名称
                    image_path = os.path.join(root, file)
                    print(image_path)
                    # 显示图片
 
                    # 排序
                    #print(predictions)
                    top_3 = predictions.argsort()[-3:][::-1]
                    for node_id in top_3:
                        # 获取分类名称
                        human_string = self.look_up(node_id)
                        # 获取该分类的置信度
                        score = predictions[node_id]
                        print('%s (score = %.5f)' % (human_string, score))
                    print()
    #加载模型方法
    def load_model(self, model_path):
        try:
            with tf.gfile.GFile(model_path, 'rb') as f:
                self.graph_def = tf.GraphDef()
                self.graph_def.ParseFromString(f.read())
                tf.import_graph_def(self.graph_def, name='')
        except Exception as ret:
            print(ret)
	#建立映射关系方法
    def load_map(self, label_path, id_path):
        # 加载target对应的分类编号字符串
        try:
            lines = tf.gfile.GFile(label_path).readlines()
        except Exception as ret:
            print(ret)
        else:
            target_map_nid = dict()
            for line in lines:
                #print(line)
                if line.startswith("  target_class:"):
                    # 分类编号
                    target_class = int(line.split(':')[1])
                if line.startswith("  target_class_string:"):
                  
                    # 提取分类编号字符串
                    #print(line.split(': ')[1])
                    target_map_nid[target_class] = line.split(': ')[1][1:-2]

        #加载字符串分类 ---对应分类名
        try:
            lines = tf.gfile.GFile(id_path).readlines()
        except Exception as ret:
            print(ret)
        else:
            label_map_proto = dict()
            for line in lines:
                #print(line)
                #去除换行符并按\t切片
                line = line.strip('\n').split('\t')
                # 获取分类编号和分类名称
                label_map_proto[line[0]] = line[1]

        #建立target与名称的映射
        target_map_name = dict()
        for key, value in target_map_nid.items():
            target_map_name[key] = label_map_proto[value]
        return target_map_name
        
     #查询id对应标签
    def look_up(self, target):
        if target not in self.__map:
            return None
        return self.__map[target]
if __name__ == '__main__':  
    x=Inception_v3()
    #x.load_model('config/classify_image_graph_def.pb')
    #x.load_map()
    x.__call__()
    #x.look_up()
    
    
    
