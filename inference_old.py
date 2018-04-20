# -*- coding: utf-8 -*-
from __future__ import division
from skimage import io,transform
import tensorflow as tf
import numpy as np
import cv2
import os
import glob

path = "./test/"
face_dict = {1:'have bladed',0:'no bladed'}

w=100
h=100
c=3

def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    print(cate)
    imgs=[]
    labels=[]
    name=[]
    for idx,folder in enumerate(cate):
        for i in glob.glob(folder+'/*.JPG'):
            print('reading the images:%s'%(i))
            name.append(i)
            im=cv2.imread(i)
            im=cv2.resize(im,(w,h),interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            labels.append(idx)
            imgs.append(img)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32),name

with tf.Session() as sess:
    data = []
    label=[]
    data,label,name = read_img(path)
    #print(data)
    saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}
    logits = graph.get_tensor_by_name("logits_eval:0")
    classification_result = sess.run(logits,feed_dict)
    #打印出预测矩阵
    #print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    #print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应叶片的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("No.",i+1,"result:"+face_dict[output[i]])
    print(label)
    print(output)
    print(name)
    for i in range(len(label)):
        if label[i]!=output[i]:
            print(i)
            print(name[i])
    #label=tf.reshape(label, [1, -1])
    #output=tf.reshape(output, [1, -1])
    #correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #print(sess.run(accuracy))
    same1=map(lambda label,output:label+output, label, output).count(2)
    same2=map(lambda label,output:label+output, label, output).count(0)
    print(same1+same2)
    accuracy=(same1+same2)/len(label)
    print("正确率是:"+str(accuracy))
