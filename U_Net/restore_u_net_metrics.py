#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:05:44 2017

@author: leilei
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.mlab as mm

def mean_IOU(pre,lab,num_class):
    HXJZ=np.zeros([num_class,num_class])
    for i in range(num_class):
        for j in range(num_class):
            Temp=pre[lab==i]
            HXJZ[i,j]=len(mm.find(Temp==j))
    return HXJZ
    

sess=tf.Session()
#import meta 结构保存 graph
new_saver=tf.train.import_meta_graph(r'/home/leilei/Ceshi/Model_fuse/model_X/model.ckpt-40000.meta')
#import weights bias 
new_saver.restore(sess,save_path=r'/home/leilei/Ceshi/Model_fuse/model_X/model.ckpt-40000')

#obtain we need
pre_2=tf.get_collection('pred_2')[0]

graph=tf.get_default_graph()

#placeholder
image=graph.get_operation_by_name('image').outputs[0]
label=graph.get_operation_by_name('label').outputs[0]
is_training=graph.get_operation_by_name('is_training').outputs[0]

img_path=r'/home/leilei/Data_X/valid/sat'
lab_path=r'/home/leilei/Data_X/valid/map'
save_path=r'/home/leilei/Ceshi/Better_model/test_pre_bn'

names=os.listdir(img_path)

for i in range(len(names)):
    img=cv2.imread(os.path.join(img_path,names[i]))
    img_list=img.tolist()
    imgs=np.array([img_list]*16)
    predict=np.array(sess.run(pre_2,feed_dict={image:imgs,is_training:False}))[0]
    p=np.float64(predict)
    p[p==1]=255
    cv2.imwrite(os.path.join(save_path,names[i]),p)

for k in range(len(names)):
    lab=cv2.imread(os.path.join(lab_path,names[k]),0)
    pred=cv2.imread(os.path.join(save_path,names[k]),0)#255
    pred[pred==255]=1
    hxjz=mean_IOU(pre=pred,lab=lab,num_class=2)
    precision=hxjz[1,1]/(hxjz[1,1]+hxjz[0,1])
    recall=hxjz[1,1]/(hxjz[1,1]+hxjz[1,0])
    mean_iou=hxjz[1,1]/(hxjz[1,1]+hxjz[0,1]+hxjz[1,0])
    print(names[k]+' '+str(precision)+' '+str(recall)+' '+str(mean_iou))

    

names=os.listdir(img_path)
num_class=2
hxjzs=np.zeros([num_class,num_class])
for k in range(len(names)):
    lab=cv2.imread(os.path.join(lab_path,names[k]),0)
    pred=cv2.imread(os.path.join(save_path,names[k]),0)#255
    pred[pred==255]=1
    hxjz=mean_IOU(pre=pred,lab=lab,num_class=num_class)
    hxjzs+=hxjz
#    precision=hxjz[1,1]/(hxjz[1,1]+hxjz[1,0])
#    recall=hxjz[1,1]/(hxjz[1,1]+hxjz[0,1])
#    mean_iou=hxjz[1,1]/(hxjz[1,1]+hxjz[0,1]+hxjz[1,0])
#    print(names[k]+' '+str(precision)+' '+str(recall)+' '+str(mean_iou))
precision=hxjzs[1,1]/(hxjzs[1,1]+hxjzs[0,1])
recall=hxjzs[1,1]/(hxjzs[1,1]+hxjzs[1,0])
mean_iou=hxjzs[1,1]/(hxjzs[1,1]+hxjzs[0,1]+hxjzs[1,0])
print(str(precision)+' '+str(recall)+' '+str(mean_iou))