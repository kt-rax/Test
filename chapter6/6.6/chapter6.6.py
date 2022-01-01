# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import logging
import time
import logging


FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}

class Inception_v3(object):
    def __init__(self,):
        #配置文件
        label_path = 'imagenet_2012_challenge_label_map_proto.pbtxt'
        id_path = 'imagenet_synset_to_human_label_map.txt'
        model_path = 'classify_image_graph_def.pb'
        self.__map = self.load_map(label_path,id_path)
        self.load_mode(model_path)
        #魔法方法call可以直接用类名加括号的方式调用该方法  
    def __call__(self,image_path = 'images/'):
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            for root,dirs,files in os.walk(image_path):
                if files is None:
                    print('directory is null')
                for file in files:
                    image_data = tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
                    predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
                    predictions = np.squeeze(predictions)
                    #打印
                    image_path = os.path.join(root,file)
                    print(image_path)
                    #显示图片 
                    img = Image.open(image_path)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
                    #排序
                    top_3 = predictions.argsort()[-3:][::-1]
                    for node_id in top_3:
                        #log系统时间，毫米级
                        logger = logging.getLogger('test time log')
                        logger.warning('' ,extra=d)
                         #获取分类名称
                        human_string = self.look_up(node_id)
                        #获取该分类的置信度 
                        score = predictions[node_id]
                        print('%s(score = %.5f)'%(human_string,score))
                    print()
                    
        #加载模型方法 
    def load_mode(self,mode_path):
        try:
            with tf.gfile.GFile(mode_path,'rb') as f:
                self.graph_def = tf.GraphDef()
                self.graph_def.ParseFromString(f.read())
                tf.import_graph_def(self.graph_def,name='')
        except Exception as ret:
            print(ret)
        
        #建立映射关系方法 
    
    def load_map(self,label_path,id_path):  
        #一共有两个映射表，1.分类号和对应的字符串编号映射，2.字符串编号和对应的字符串的映射 
        #（1）加载target对应的分类编号字符串 
        try: 
            lines = tf.gfile.GFile(label_path).readlines()
        except Exception as ret:
            print(ret)
        else:
            target_map_nid = dict()
            
            for line in lines:
                if line.startswith("  target_class:"):
                    #分类编号
                    target_class = int(line.split(':')[1])
                
                if line.startswith("  target_class_string:"):
                    #提取分类编号字符串 
                    target_map_nid[target_class] = line.split(': ')[1][1:-2]
                '''     
                if line.startswith("  target_class_string:"):
                    target_map_nid[target_class] = line.split(': ')[1][1:-2]
                '''
                             
        #（2）加载字符串分类---对应分类名

        #加载字符串分类 ---对应分类名
        try:
            lines = tf.gfile.GFile(id_path).readlines()
        except Exception as ret:
            print(ret)
        else:
            label_map_proto = dict()
            for line in lines:
                #去除换行符\t切片 
                line = line.strip('\n').split('\t')
                #获取分类编号与分类名称
                label_map_proto[line[0]] = line[1]
                #建立target与名称的映射 
        target_map_name = dict()
        for key,value in target_map_nid.items():
            target_map_name[key] = label_map_proto[value]
        return target_map_name               
        
        #查询id对应标签

    def look_up(self,target):
        if target not in self.__map:
            return None
        return self.__map[target]


if __name__ == '__main__':
    x = Inception_v3()
    x.__call__()

        
        

