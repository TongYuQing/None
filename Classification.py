#机器学习课程第一次作业
#
#
import numpy as np
from sklearn import datasets
digits = datasets.load_digits()
#输出数据集的样本数与特征数
print("数据集的样本数与特征数为：")
print(digits.data.shape)
#输出所有目标类别
print("所有目标类别:")
print(np.unique(digits.target))

#############################################################
#数据集可视化
import matplotlib.pyplot as plt
# 导入字体管理器，用于提供中文支持
import matplotlib.font_manager as fm
font_set= fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc', size=14)

# 将图像和目标标签合并到一个列表中
images_and_labels = list(zip(digits.images, digits.target))

# 打印数据集的前8个图像
plt.figure(figsize=(8,6))
for index, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title(u'样本：' + str(label), fontproperties=font_set)
    
plt.show()

############################################################
#用PCA降维
from sklearn.decomposition import *
#创建一个pca模型
pca = PCA(n_components=2)

#将数据应用到模型上
reduced_data_pca = pca.fit_transform(digits.data)

# 查看维度
print(reduced_data_pca.shape)

#绘制散点图
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
plt.figure(figsize=(8, 6))
for i in range(len(colors)):
    x = reduced_data_pca[:, 0][digits.target == i]
    y = reduced_data_pca[:, 1][digits.target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel(u'第一个主成分', fontproperties=font_set)
plt.ylabel(u'第二个主成分', fontproperties=font_set)
plt.title(u"PCA 散点图", fontproperties=font_set)
plt.show()

#############################################################
#数据归一化
from sklearn.preprocessing import scale

data = scale(digits.data)
#########################################
#将数据集拆分成训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, images_train, images_test = \
train_test_split(data, digits.target, digits.images, \
                 test_size=0.25, random_state=42)
print("训练集", X_train.shape)
print("测试集", X_test.shape)

################################################################
#模型构建，使用SVM分类器
from sklearn import svm
#创建SVC模型
svc_model = svm.SVC(gamma=0.001,C=100,kernel='linear')

# 将训练集应用到 SVC 模型上
svc_model.fit(X_train, y_train)

# 评估模型的预测效果
print(svc_model.score(X_test, y_test))


#优化参数
svc_model = svm.SVC(gamma=0.001, C=10, kernel='rbf')

svc_model.fit(X_train, y_train)

print(svc_model.score(X_test, y_test))

###############################################################

#预测结果
import matplotlib.pyplot as plt
#使用创建的 SVC 模型对测试集进行预测
predicted = svc_model.predict(X_test)

# 将测试集的图像与预测的标签合并到一个列表中
images_and_predictions = list(zip(images_test, predicted))

# 打印前 4 个预测的图像和结果
plt.figure(figsize=(8, 2))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(1, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'预测结果: ' + str(prediction), fontproperties=font_set)

plt.show()

##############################################################
#分析结果的准确性
X = np.arange(len(y_test))
# 生成比较列表，如果预测的结果正确，则对应位置为0，错误则为1
comp = [0 if y1 == y2 else 1 for y1, y2 in zip(y_test, predicted)]
plt.figure(figsize=(8, 6))
# 图像发生波动的地方，说明预测的结果有误
plt.plot(X, comp)
plt.ylim(-1, 2)
plt.yticks([])
plt.show()

print("测试集数量：", len(y_test))
print("错误识别数：", sum(comp))
print("识别准确率：", 1 - float(sum(comp)) / len(y_test))

#################################################################
#错误识别样本分析
# 收集错误识别的样本下标
wrong_index = []
for i, value in enumerate(comp):
    if value: wrong_index.append(i)
    
    
# 输出错误识别的样本图像
plt.figure(figsize=(8, 6))
for plot_index, image_index in enumerate(wrong_index):
    image = images_test[image_index]
    plt.subplot(2, 4, plot_index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # 图像说明，8->9 表示正确值为8，被错误地识别成了9
    info = "{right}->{wrong}".format(right=y_test[image_index], wrong=predicted[image_index])
    plt.title(info, fontsize=16)
    
plt.show()

'''
#多层感知机
from sklearn import neural_network
import numpy as np
import sys

mlp = neural_network.MLPClassifier(hidden_layer_sizes=(40,40,40), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200 ,tol=1e-4)

mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))
#print(mlp.n_outputs_)#输出类别数
#print(mlp.classes_)#所有类别
print(mlp.loss_)#损失函数的损失值
#print(mlp.intercepts_)#偏移量
#print(mlp.coefs_)#权重
print(mlp.n_iter_)#迭代轮数
print(mlp.n_layers_)#网络层数，只有一层隐藏层时 =3
print(mlp.out_activation_)#输出层激活函数的名称
'''
