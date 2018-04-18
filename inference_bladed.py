# -*- coding: utf-8 -*-

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
    for idx,folder in enumerate(cate):
        for i in glob.glob(folder+'/*.JPG'):
            print('reading the images:%s'%(i))
            im=cv2.imread(i)
            im=cv2.resize(im,(w,h),interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            labels.append(idx)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像
            edges = cv2.Canny(gray,30,200)
            #plt.subplot(121),plt.imshow(edges,'gray')
            #创建一个空画布hough_channel
            hough_channel = np.zeros(img.shape, np.uint8)
            hough_channel=cv2.cvtColor(hough_channel,cv2.COLOR_BGR2GRAY)
            #print(hough_channel.shape)
            #print(img.shape)
            #hough transform
            lines = cv2.HoughLines(edges,1,np.pi/180,130)
            try:
                lines1 = lines[:,0,:]#提取为为二维
                for rho,theta in lines1[:]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(hough_channel,(x1,y1),(x2,y2),(255,255,255),1)
            except Exception as e:
                print 'There is no lines to be detected!'
            #Sobel边缘检测
            sobelX = cv2.Sobel(gray,cv2.CV_64F,1,0)#x方向的梯度
            sobelY = cv2.Sobel(gray,cv2.CV_64F,0,1)#y方向的梯度
            sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
            sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值

            #merge
            mergedByNp = np.dstack([img,hough_channel,sobelX,sobelY]) #生成7通道
            imgs.append(mergedByNp)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

with tf.Session() as sess:
    data = []
    data,label = read_img(path)
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
    label=tf.reshape(label, [1, -1])
    output=tf.reshape(output, [1, -1])
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("正确率是："+sess.run(accuracy))
