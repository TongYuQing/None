#机器学习编程作业三，仅适用于数据量较小的情况
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np 
import time 

img_path = '/home/tong/图片/Wallpapers'

# 将所有图片resize成 w * h
w = 128
h = 128
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
def  minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
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
y_ = tf.placeholder(tf.int32, shape=[None,], name='y_')


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

logits = CNNlayer()
loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#选取多次迭代中测试集精度最高的三次进行保存
saver=tf.train.Saver(max_to_keep=3)

max_acc=0
f=open('/home/tong/图片/Wallpapers/acc.txt', 'w')
 
 
n_epoch = 10
batch_size = 64
sess=tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()
    
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (train_loss/ n_batch))
    print("   train acc: %f" % (train_acc/ n_batch))
    
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (val_loss/ n_batch))
    print("   validation acc: %f" % (val_acc/ n_batch))
 
    f.write(str(epoch+1)+', val_acc: '+str(val_acc)+'\n')
    if val_acc>max_acc:
        max_acc=val_acc
        saver.save(sess,'/home/tong/faces.ckpt',global_step=epoch+1)
 
f.close()
sess.close()
