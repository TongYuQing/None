from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np 
import time 

img_path = '/home/tong/图片/Wallpapers'

# 将所有图片resize成 w * h
w = 112
h = 96
c = 3

# 读取图片
def  read_img(img_path):
    subject_lst = [img_path + '/' + x for x in os.listdir(img_path) \
                                        if os.path.isdir(img_path + '/' + x)]
    imgs = []
    labels = []
    for idx,folder in enumerate(subject_lst):
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images: %s' % (im))
            img = io.imread(im)
            img = transform.resize(img,(w,h,c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

data, label = read_img(img_path)

# 打乱图片顺序
num_img = data.shape[0]
arr = np.arange(num_img)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# 将所有数据分为训练集和验证集
ratio = 0.8
a = np.int(num_img * ratio)
x_train = data[:a]
y_train = label[:a]
x_val = data[a:]
y_val = label[a:]

# 批量读取数据进行训练
def  minbatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(targets) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#-----------------------开始构建网络------------------------------------------
#
#
x = tf.placeholder(tf.float32,shape=[None,w,h,c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None,], neme='y_')


def CNNlayer():
    #第一个卷积层（128——>64)
    conv1 =tf.layers.conv2d(
          inputs=x,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
 
    #第二个卷积层(64->32)
    conv2=tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
 
    #第三个卷积层(32->16)
    conv3=tf.layers.conv2d(
          inputs=pool2,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
 
    #第四个卷积层(16->8)
    conv4=tf.layers.conv2d(
          inputs=pool3,
          filters=128,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
 
    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])
 
    #全连接层
    dense1 = tf.layers.dense(inputs=re1, 
                          units=1024, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2= tf.layers.dense(inputs=dense1, 
                          units=512, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits= tf.layers.dense(inputs=dense2, 
                            units=60,
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return logits
