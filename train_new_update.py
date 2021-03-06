# -*- coding: utf-8 -*-

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

#数据集地址
path='./dataset/'
#模型保存地址
model_path='./new_model_update/model.ckpt'
#图片resize参数
w=100
h=100
c=6
#读取图片
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    print(cate)
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for i in glob.glob(folder+'/*.JPG'):
            print('reading the images:%s'%(i))
            im=cv2.imread(i)
            im=cv2.resize(im,(200,200),interpolation=cv2.INTER_CUBIC)
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
            lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength=20,maxLineGap=0.8)
            try:
                lines1 = lines[:,0,:]#提取为二维
                for x1,y1,x2,y2 in lines1[:]:
                    cv2.line(hough_channel,(x1,y1),(x2,y2),(255,255,255),1)
            except Exception as e:
                print 'There is no lines to be detected!'
            #Sobel边缘检测
            sobelX = cv2.Sobel(gray,cv2.CV_64F,1,0)#x方向的梯度
            sobelY = cv2.Sobel(gray,cv2.CV_64F,0,1)#y方向的梯度
            sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
            sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值
            #resize
            img=cv2.resize(img,(100,100),interpolation=cv2.INTER_CUBIC)
            hough_channel=cv2.resize(hough_channel,(100,100),interpolation=cv2.INTER_CUBIC)
            sobelX=cv2.resize(sobelX,(100,100),interpolation=cv2.INTER_CUBIC)
            sobelY=cv2.resize(sobelY,(100,100),interpolation=cv2.INTER_CUBIC)
            #savefig
            #plt.subplot(151),plt.imshow(img,),plt.title('RGB')
            #plt.subplot(152),plt.imshow(hough_channel,),plt.title('hough')
            #plt.subplot(153),plt.imshow(sobelX,),plt.title('x')
            #plt.subplot(154),plt.imshow(sobelY,),plt.title('y')
            #plt.show()
            #plt.savefig("/home/jzx/working/cNN_BLADED/savefig/"+str(i)[2:-4]+'.jpg')

            #merge
            mergedByNp = np.dstack([img,hough_channel,sobelX,sobelY]) #生成6通道
            imgs.append(mergedByNp)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

data,label=read_img(path)
#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
print(data.shape)
label=label[arr]
#将所有数据分为训练集和验证集
ratio=0.8
s1=np.int(num_example*ratio)
s2=np.int(num_example*0.95)
x_train=data[:s1]
y_train=label[:s1]
x_val=data[s1:s2]
y_val=label[s1:s2]
x_test=data[s2:]
y_test=label[s2:]


#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
keep_prob = tf.placeholder(tf.float32)

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[3,3,6,6],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv1_biases = tf.get_variable("bias", [6], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope("layer2-conv2"):
        conv2_weights = tf.get_variable("weight",[3,3,6,6],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv2_biases = tf.get_variable("bias", [6], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer3-pool1"):
        pool1 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75,name='norm1')

    with tf.variable_scope("layer4-conv3"):
        conv3_weights = tf.get_variable("weight",[3,3,6,12],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv3_biases = tf.get_variable("bias", [12], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(norm1, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.variable_scope("layer5-conv4"):
        conv4_weights = tf.get_variable("weight",[3,3,12,12],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv4_biases = tf.get_variable("bias", [12], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
    with tf.name_scope("layer6-pool2"):
        pool2 = tf.nn.max_pool(relu4, ksize = [1,2,2,1],strides=[1,2,2,1],padding='VALID')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75,name='norm2')

    with tf.variable_scope('layer7-conv5'):
        conv5_weights = tf.get_variable("weight",[3,3,12,24],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv5_biases = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(norm2, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
    with tf.variable_scope("layer8-conv6"):
        conv6_weights = tf.get_variable("weight",[3,3,24,24],initializer=tf.truncated_normal_initializer(stddev=1.0))
        conv6_biases = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(relu5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))
    with tf.name_scope("layer9-pool3"):
        pool3 = tf.nn.max_pool(relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        nodes=13*13*24
        reshaped = tf.reshape(pool3,[-1,nodes])

    with tf.variable_scope('layer10-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer11-fc2'):
        fc2_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases


    return logit

#---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x,False,regularizer)

#将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)

train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#训练和测试数据，可将n_epoch和batch_size
n_epoch=30
batch_size=64
saver=tf.train.Saver()
#GPU和cpu选择
config = tf.ConfigProto(device_count = {'GPU': 1}) #0表示仅cpu，1表示使用gpu
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#tensorboard  tensorboard --logdir=./log
summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log",sess.graph)
writer.add_graph(sess.graph)

#sess=tf.Session()

sess.run(tf.global_variables_initializer())

for epoch in range(n_epoch):
    start_time = time.time()
    #print(x.shape)
    print("====epoch %d====="%epoch)
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _=sess.run([train_op],feed_dict={x: x_train_a, y_: y_train_a, keep_prob: 0.5})
        err,ac=sess.run([loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=True):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))

        #test
    test_acc = sess.run(acc,feed_dict={x:x_test,y_:y_test,keep_prob:1.0})
    print("   test acc: %f" % (test_acc))


#保存模型，后期调用
saver.save(sess,model_path)

writer.close()
sess.close()
