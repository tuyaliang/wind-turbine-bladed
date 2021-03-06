#coding=utf-8
#input:240*320*7
#最后一层pooling输出：60×80×64

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
model_path='./model/model.ckpt'
#图片resize参数
w=240
h=320
c=7
#图像预处理DoG处理函数
def  getExtrema(A, B, C, thresh):
    height,width= A.shape
    resu = np.ones((height, width), A.dtype) * 100
    for row in range(1, height-1):
        for col in range(1, width-1):
            center = B[row, col]
            if center < thresh:
                continue
            B[row, col] = B[row, col - 1]
            minValue = np.vstack([A[row-1:row+2, col-1:col+2], B[row-1:row+2, col-1:col+2],C[row-1:row+2, col-1:col+2]]).min()
            maxValue = np.vstack([A[row - 1:row + 2, col - 1:col + 2], B[row - 1:row + 2, col - 1:col + 2],
                                  C[row - 1:row + 2, col - 1:col + 2]]).max()
            if center < minValue:
                resu[row, col] = 0
            if center > maxValue:
                resu[row, col] = 255
            B[row, col] = center
    return resu

def addPoint(image, image_point):
    #height, width, dvim = image.shape
    height, width = image.shape
    for row in range(0, height):
        for col in range(0, width):
            if image_point[row, col] == 255:
                cv2.circle(image, (row, col), 5, thickness=1, color=[255,255,255])
            elif image_point[row, col] == 0:
                cv2.circle(image, (row, col), 5, thickness=1, color=[0,0,0])

#读取图片
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    print(cate)
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for i in glob.glob(folder+'/*.jpg'):
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
            lines = cv2.HoughLines(edges,1,np.pi/180,160)
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
            #dog处理
            r,g,b = cv2.split(im)
            #创建一个空画布dog_channel
            dog_channel = np.zeros(im.shape, np.uint8)
            dog_channel=cv2.cvtColor(dog_channel,cv2.COLOR_BGR2GRAY)
            image_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            image_gray_blur1 = cv2.GaussianBlur(gray, (3, 3), 0.1)
            image_gray_blur2 = cv2.GaussianBlur(gray, (3, 3), 0.2)
            image_gray_blur3 = cv2.GaussianBlur(gray, (3, 3), 0.3)
            image_gray_blur4 = cv2.GaussianBlur(gray, (3, 3), 0.4)
            image_gray_blur5 = cv2.GaussianBlur(gray, (3, 3), 0.5)
            image_gray_blur6 = cv2.GaussianBlur(gray, (3, 3), 0.6)
            image_gray_dog1 = image_gray_blur2 - image_gray_blur1
            image_gray_dog2 = image_gray_blur4 - image_gray_blur3
            image_gray_dog3 = image_gray_blur6 - image_gray_blur5
            image_point = getExtrema(image_gray_dog1, image_gray_dog2, image_gray_dog3, 2)
            addPoint(dog_channel, image_point)
            #merge
            mergedByNp = np.dstack([img,hough_channel,sobelX,sobelY,dog_channel]) #生成7通道
            imgs.append(mergedByNp)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

data,label=read_img(path)
#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
#print(data.shape)
label=label[arr]
#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
#print(x_train.shape)
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]
#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[5,5,7,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        tf.summary.image("conv1",conv1,3)

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        tf.summary.image("pool1",pool1,3)
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        tf.summary.image("conv2",conv2,3)
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 60*80*64
        reshaped = tf.reshape(pool2,[-1,nodes])
        tf.summary.image("pool2",pool2,3)
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit

#---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x,False,regularizer)

#将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
tf.summary.scalar("loss",loss)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("acc",acc)
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
config = tf.ConfigProto(device_count = {'GPU': 0}) #0表示仅cpu，1表示使用gpu
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#tensorboard  tensorboard --logdir=./log
merged_summary = tf.summary.merge_all()
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
        summary,_,err,ac=sess.run([merged_summary,train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
        writer.add_summary(summary,epoch)
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1

    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))


#保存模型，后期调用
saver.save(sess,model_path)

writer.close()
sess.close()
