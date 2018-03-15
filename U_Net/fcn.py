#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:57:49 2017

@author: leilei
"""

import tensorflow as tf
from math import ceil
import numpy as np
from tensorflow.contrib.layers import xavier_initializer

class Model:
    ''' all layers need to be initializer '''
    def __init__(self,dataset_mean_vector):
        self.epsilon = tf.constant(value=1e-4)
        self.mean_vector=dataset_mean_vector
        
    def conv(self,bottom,ksize,is_training,name):
        ''' conv layer bn then activation '''
        with tf.variable_scope(name):
            weights=tf.get_variable('weights',shape=ksize,dtype=tf.float32,initializer=xavier_initializer())
            biases=tf.get_variable('biases',shape=[ksize[-1]],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            output=tf.nn.conv2d(bottom,filter=weights,strides=[1,1,1,1],padding='SAME')
            output=tf.nn.bias_add(output,biases)
            #bn
            output=tf.contrib.layers.batch_norm(output,center=True,scale=True,is_training=is_training)
            output=tf.nn.relu(output)
            return output
            
    def max_pool(self,bottom,name):
        with tf.variable_scope(name):
            output=tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)
            return output
    
    def upsample(self,bottom,shape,num_outputs,ksize,stride,name):
        ''' initialize upsample '''
        with tf.variable_scope(name):
            strides=[1,stride,stride,1]
            num_filters_in=bottom.get_shape()[-1].value
            kernel_shape=[ksize,ksize,num_outputs,num_filters_in]
            output_shape=tf.stack([shape[0],shape[1],shape[2],num_outputs])
            weights=tf.get_variable('weights',kernel_shape,tf.float32,xavier_initializer())
            upsample=tf.nn.conv2d_transpose(bottom,filter=weights,output_shape=output_shape,
                                            strides=strides,padding='SAME',name=name)
            return upsample
            
    
    def deconv(self,bottom,shape,num_outputs,ksize,stride,name):
        ''' bilinear interpolation '''
        with tf.variable_scope(name):
            strides=[1,stride,stride,1]
            num_filters_in=bottom.get_shape()[-1].value
            if shape is None:
                in_shape=tf.shape(bottom)
                h=((in_shape[1]-1)*stride)+1
                w=((in_shape[2]-1)*stride)+1
                new_shape=[in_shape[0],h,w,num_outputs]
            else:
                new_shape=[shape[0],shape[1],shape[2],num_outputs]
            output_shape=tf.stack(new_shape)
            weights_shape=[ksize,ksize,num_outputs,num_filters_in]
            weights=self.deconv_weight_variable(weights_shape)
            deconv=tf.nn.conv2d_transpose(bottom,filter=weights,output_shape=output_shape,
                                          strides=strides,padding='SAME',name=name)
            return deconv
            
    def deconv_weight_variable(self,weights_shape):
        height=weights_shape[0]
        width=weights_shape[1]
        f=ceil(height/2.0)
        c=(2*f-1-f%2)/(2.0*f)
        bilinear=np.zeros([height,width])
        for y in range(height):
            for x in range(width):
                value=(1-abs(y/f-c))*(1-abs(x/f-c))
                bilinear[y,x]=value
        weights=np.zeros(weights_shape)
        for i in range(weights_shape[2]):
            weights[:,:,i,i]=bilinear
            ''' unknown '''
        init=tf.constant_initializer(value=weights)
        return tf.get_variable('up_filter',shape=weights.shape,dtype=tf.float32,initializer=init)
    
    def build(self,input_image,class_number,is_training):
        with tf.name_scope('processing'):
            b,g,r=tf.split(input_image,3,axis=3)
            input_image_=tf.concat([
                    b*0.00390625,
                    g*0.00390625,
                    r*0.00390625],axis=3)
            
#            input_image_=tf.concat([
#                    b-self.mean_vector[0],
#                    g-self.mean_vector[1],
#                    r-self.mean_vector[2]],axis=3)
        '''
        HF_FCN
        '''
#        self.conv1_1=self.conv(input_image_,ksize=[3,3,input_image_.get_shape()[-1].value,64],is_training=is_training,name='conv1_1')
#        self.conv1_2=self.conv(self.conv1_1,ksize=[3,3,64,64],is_training=is_training,name='conv1_2')
#        self.pool1=self.max_pool(self.conv1_2,name='pool1')
#        
#        self.conv2_1=self.conv(self.pool1,ksize=[3,3,64,128],is_training=is_training,name='conv2_1')
#        self.conv2_2=self.conv(self.conv2_1,ksize=[3,3,128,128],is_training=is_training,name='conv2_2')
#        self.pool2=self.max_pool(self.conv2_2,name='pool2')
#        
#        self.conv3_1=self.conv(self.pool2,ksize=[3,3,128,256],is_training=is_training,name='conv3_1')
#        self.conv3_2=self.conv(self.conv3_1,ksize=[3,3,256,256],is_training=is_training,name='conv3_2')
#        self.conv3_3=self.conv(self.conv3_2,ksize=[3,3,256,256],is_training=is_training,name='conv3_3')
#        self.pool3=self.max_pool(self.conv3_3,name='pool3')
#        
#        self.conv4_1=self.conv(self.pool3,ksize=[3,3,256,512],is_training=is_training,name='conv4_1')
#        self.conv4_2=self.conv(self.conv4_1,ksize=[3,3,512,512],is_training=is_training,name='conv4_2')
#        self.conv4_3=self.conv(self.conv4_2,ksize=[3,3,512,512],is_training=is_training,name='conv4_3')
#        self.pool4=self.max_pool(self.conv4_3,name='pool4')
#        
#        self.conv5_1=self.conv(self.pool4,ksize=[3,3,512,512],is_training=is_training,name='conv5_1')
#        self.conv5_2=self.conv(self.conv5_1,ksize=[3,3,512,512],is_training=is_training,name='conv5_2')
#        self.conv5_3=self.conv(self.conv5_2,ksize=[3,3,512,512],is_training=is_training,name='conv5_3')
#        
#        # DSN conv1_1
#        self.upscore_dsn1_1=self.conv(self.conv1_1,ksize=[1,1,64,1],is_training=is_training,name='upscore_dsn1_1')
#        # DSN conv1_2
#        self.upscore_dsn1_2=self.conv(self.conv1_2,ksize=[1,1,64,1],is_training=is_training,name='upscore_dsn1_2')
#        # DSN conv2_1
#        self.score_dsn2_1=self.conv(self.conv2_1,ksize=[1,1,128,1],is_training=is_training,name='score_dsn2_1')
#        self.upscore_dsn2_1=self.deconv(self.score_dsn2_1,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=4,stride=2,name='upscore_dsn2_1')
#        # DSN conv2_2
#        self.score_dsn2_2=self.conv(self.conv2_2,ksize=[1,1,128,1],is_training=is_training,name='score_dsn2_2')
#        self.upscore_dsn2_2=self.deconv(self.score_dsn2_2,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=4,stride=2,name='upscore_dsn2_2')
#        # DSN conv3_1
#        self.score_dsn3_1=self.conv(self.conv3_1,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_1')
#        self.upscore_dsn3_1=self.deconv(self.score_dsn3_1,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=8,stride=4,name='upscore_dsn3_1')
#        # DSN conv3_2
#        self.score_dsn3_2=self.conv(self.conv3_2,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_2')
#        self.upscore_dsn3_2=self.deconv(self.score_dsn3_2,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=8,stride=4,name='upscore_dsn3_2')
#        # DSN conv3_3
#        self.score_dsn3_3=self.conv(self.conv3_3,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_3')
#        self.upscore_dsn3_3=self.deconv(self.score_dsn3_3,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=8,stride=4,name='upscore_dsn3_3')
#        # DSN conv4_1
#        self.score_dsn4_1=self.conv(self.conv4_1,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_1')
#        self.upscore_dsn4_1=self.deconv(self.score_dsn4_1,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=16,stride=8,name='upscore_dsn4_1')
#        # DSN conv4_2
#        self.score_dsn4_2=self.conv(self.conv4_2,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_2')
#        self.upscore_dsn4_2=self.deconv(self.score_dsn4_2,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=16,stride=8,name='upscore_dsn4_2')
#        # DSN conv4_3
#        self.score_dsn4_3=self.conv(self.conv4_3,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_3')
#        self.upscore_dsn4_3=self.deconv(self.score_dsn4_3,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=16,stride=8,name='upscore_dsn4_3')
#        # DSN conv5_1
#        self.score_dsn5_1=self.conv(self.conv5_1,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_1')
#        self.upscore_dsn5_1=self.deconv(self.score_dsn5_1,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=32,stride=16,name='upscore_dsn5_1')
#        # DSN conv5_2
#        self.score_dsn5_2=self.conv(self.conv5_2,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_2')
#        self.upscore_dsn5_2=self.deconv(self.score_dsn5_2,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=32,stride=16,name='upscore_dsn5_2')
#        # DSN conv5_3
#        self.score_dsn5_3=self.conv(self.conv5_3,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_3')
#        self.upscore_dsn5_3=self.deconv(self.score_dsn5_3,shape=input_image_.get_shape().as_list(),num_outputs=1,ksize=32,stride=16,name='upscore_dsn5_3')
#        # Concat
#        self.Concat_sum=tf.concat([self.upscore_dsn1_1,self.upscore_dsn1_2,self.upscore_dsn2_1,self.upscore_dsn2_2,self.upscore_dsn3_1,self.upscore_dsn3_2,self.upscore_dsn3_3,
#                                   self.upscore_dsn4_1,self.upscore_dsn4_2,self.upscore_dsn4_3,self.upscore_dsn5_1,self.upscore_dsn5_2,self.upscore_dsn5_3],axis=3)
#        
#        self.new_score_1=self.conv(self.Concat_sum,ksize=[1,1,self.Concat_sum.get_shape()[3].value,class_number],is_training=False,name='new_score_1')
#        
#        self.softmax_1=tf.nn.softmax(self.new_score_1+self.epsilon)
#        self.pred_1=tf.argmax(self.softmax_1,axis=3)
        
        
        '''
        U_Net : no dropout
        '''
        self.conv_1=self.conv(input_image_,ksize=[3,3,input_image_.get_shape()[-1].value,32],is_training=is_training,name='conv_1')
        self.conv_2=self.conv(self.conv_1,ksize=[3,3,32,32],is_training=is_training,name='conv_2')
        self.pool_1=self.max_pool(self.conv_2,name='pool_1')
        
        self.conv_3=self.conv(self.pool_1,ksize=[3,3,32,64],is_training=is_training,name='conv_3')
        self.conv_4=self.conv(self.conv_3,ksize=[3,3,64,64],is_training=is_training,name='conv_4')
        self.pool_2=self.max_pool(self.conv_4,name='pool_2')
        
        self.conv_5=self.conv(self.pool_2,ksize=[3,3,64,128],is_training=is_training,name='conv_5')
        self.conv_6=self.conv(self.conv_5,ksize=[3,3,128,128],is_training=is_training,name='conv_6')
        self.pool_3=self.max_pool(self.conv_6,name='pool_3')
        
        self.conv_7=self.conv(self.pool_3,ksize=[3,3,128,256],is_training=is_training,name='conv_7')
        self.conv_8=self.conv(self.conv_7,ksize=[3,3,256,256],is_training=is_training,name='conv_8')
        self.pool_4=self.max_pool(self.conv_8,name='pool_4')
        
        self.conv_9=self.conv(self.pool_4,ksize=[3,3,256,512],is_training=is_training,name='conv_9')
        self.conv_10=self.conv(self.conv_9,ksize=[3,3,512,512],is_training=is_training,name='conv_10')
        
        self.up_10=self.upsample(self.conv_10,shape=self.conv_8.get_shape().as_list(),num_outputs=256,ksize=2,stride=2,name='up_10')
        self.concat_10_8=tf.concat([self.up_10,self.conv_8],axis=3)
        
        self.conv_11=self.conv(self.concat_10_8,ksize=[3,3,self.concat_10_8.get_shape()[-1].value,256],is_training=is_training,name='conv_11')
        self.conv_12=self.conv(self.conv_11,ksize=[3,3,256,256],is_training=is_training,name='conv_12')
        
        self.up_12=self.upsample(self.conv_12,shape=self.conv_6.get_shape().as_list(),num_outputs=128,ksize=2,stride=2,name='up_12')
        self.concat_12_6=tf.concat([self.up_12,self.conv_6],axis=3)
        
        self.conv_13=self.conv(self.concat_12_6,ksize=[3,3,self.concat_12_6.get_shape()[-1].value,128],is_training=is_training,name='conv_13')
        self.conv_14=self.conv(self.conv_13,ksize=[3,3,128,128],is_training=is_training,name='conv_14')
        
        self.up_14=self.upsample(self.conv_14,shape=self.conv_4.get_shape().as_list(),num_outputs=64,ksize=2,stride=2,name='up_14')
        self.concat_14_4=tf.concat([self.up_14,self.conv_4],axis=3)
        
        self.conv_15=self.conv(self.concat_14_4,ksize=[3,3,self.concat_14_4.get_shape()[-1].value,64],is_training=is_training,name='conv_15')
        self.conv_16=self.conv(self.conv_15,ksize=[3,3,64,64],is_training=is_training,name='conv_16')
        
        self.up_16=self.upsample(self.conv_16,shape=self.conv_2.get_shape().as_list(),num_outputs=32,ksize=2,stride=2,name='up_16')
        self.concat_16_2=tf.concat([self.up_16,self.conv_2],axis=3)
        
        self.conv_17=self.conv(self.concat_16_2,ksize=[3,3,self.concat_16_2.get_shape()[-1].value,32],is_training=is_training,name='conv_17')
        self.conv_18=self.conv(self.conv_17,ksize=[3,3,32,32],is_training=is_training,name='conv_18')
        
        self.new_score_2=self.conv(self.conv_18,ksize=[1,1,32,class_number],is_training=False,name='new_score_2')
        
        self.softmax_2=tf.nn.softmax(self.new_score_2+self.epsilon)
        self.pred_2=tf.argmax(self.softmax_2,axis=3)
        
        # model fuse
#        self.softmax=tf.add(self.softmax_1,self.softmax_2)
#        self.pred=tf.argmax(self.softmax,axis=3)






