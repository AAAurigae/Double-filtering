import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
import math

#数据集，电影的评论，被分为positive and
imdb = keras.datasets.imdb
word_index = imdb.get_word_index()
vocab_size = len(word_index)+4
index_from = 3

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words = vocab_size,index_from = index_from)

word_index_1 = {k:(v+3) for k, v in word_index.items()}#把所有的id往上偏移 3
#几个特殊字符
word_index_1['<PAD>'] = 0
word_index_1['START'] = 1
word_index_1['UNK'] = 2
word_index_1['END'] = 3

text_len_li = list(map(len, train_data))
#对数据进行补全（padding）
max_length = 500 #句子长度低于500的句子会被补全，高于500 的会被截断
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,#list of list
    value = word_index_1['<PAD>'],
    padding = 'post',#post是把padding安在句子的后面，pre是把padding安在句子的前面
    maxlen = max_length
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,#list of list
    value = word_index_1['<PAD>'],
    padding = 'post',#post是把padding安在句子的后面，pre是把padding安在句子的前面
    maxlen = max_length
)


#样本矩阵
M = np.empty([25000,vocab_size], dtype = int)
for j in range(vocab_size):
    for i in range(25000):
        M[i][j] = np.sum(train_data[i]==j)

M_1 = np.delete(M,[0,1,2,3],axis=1)

#特征筛选
def MV(X, Y):
    n = X.shape[0]  # 行数
    p = X.shape[1]  # 列数
    pr = np.zeros(2)
    n_1 = len(np.where(Y == 1)[0])  # 找到Y中值为1的个数
    n_2 = n - n_1
    pr[0] = n_1 / n
    pr[1] = n_2 / n
    mv = []
    for j in range(p):
        F1 = [[1 if bb < aa else 0 for bb in X[np.where(Y == 1), j][0]] for aa in X[:, j].T]
        # 找到Y中值为1的相对应的X[:,j]列的值，与X[:,j]中每个值做比较
        F2 = [[1 if bb < aa else 0 for bb in X[np.where(Y == 0), j][0]] for aa in X[:, j].T]
        # 找到Y中值为2的相对应的X[:,j]列的值，与X[:,j]中每个值做比较
        FF = [[1 if bb < aa else 0 for bb in X[:, j]] for aa in X[:, j].T]
        # 将X[:,j]中每个值互相做比较

        F1_apply = np.apply_along_axis(lambda s: sum(s), 1, np.array(F1)) / n / pr[0]
        # 每列应用求和函数
        FF_apply = np.apply_along_axis(lambda s: sum(s), 1, np.array(FF))
        F1_apply_sub_FF_apply = pr[0] * ((F1_apply - FF_apply) ** 2)

        F2_apply = np.apply_along_axis(lambda s: sum(s), 1, np.array(F2)) / n / pr[1]
        FF_apply = np.apply_along_axis(lambda s: sum(s), 1, np.array(FF))
        F2_apply_sub_FF_apply = pr[1] * ((F2_apply - FF_apply) ** 2)

        mv.append(sum(F1_apply_sub_FF_apply + F2_apply_sub_FF_apply) / n)
    return (p - 1) - np.argsort(mv)


Y = train_labels[0:25000]
X = M_1
d = 200
data_full = np.hstack((np.mat(Y).T,X))#水平方向合并矩阵Y.T与X
data_full1 = [i.tolist() for i in data_full.A if i[0]==0]
#找到data_full中Y值为1的行
data_full2 = [i.tolist() for i in data_full.A if i[0]==1]
#找到data_full中Y值为2的行
outdata = np.array(data_full1+data_full2)#按照行合并data_full1和data_full2
Y_full = outdata[:,0]#第0列为Y
X_full = outdata[:,1:(vocab_size-3)]#1到p列为X
MV_1 = MV(X_full,Y_full)

#新指标字典
word_index_f={}
word_index_new={}
key_list = []
value_list = []
for key, value in word_index.items():
    key_list.append(key)
    value_list.append(value)
for get_value in range(vocab_size-4):
    get_value_index = value_list.index(get_value+1)
    word_index_f[key_list[get_value_index]] = np.where(MV_1 == get_value)[0][0] + 1
    word_index_new[key_list[get_value_index]] = get_value + 1 + np.where(MV_1 == get_value)[0][0]+1

#排序sort
word_index_order=sorted(word_index.items(),key=lambda x:x[1],reverse=False)
word_index_f_order=sorted(word_index_f.items(),key=lambda x:x[1],reverse=False)
word_index_new_order=sorted(word_index_new.items(),key=lambda x:x[1],reverse=False)

#保留指标小于500的单词，记为新字典
word_index_new_filter = filter(lambda item: item[1] <= 500, word_index_new.items())
word_index_new_filter =dict(word_index_new_filter)
word_index_new_filter_order=sorted(word_index_new_filter.items(),key=lambda x:x[1],reverse=False)

word_all = list(word_index.values())
key_new = list(word_index_new_filter.keys())
new_word_value = []

for i in key_new:
    new_word_value.append(word_index[key_new[i]])

for i in new_word_value:
    word_all.remove(new_word_value[i])

for i in range(25000):
    train_data[i] = np.delete(train_data[i],word_all)

#定义模型
embedding_dim = 16#每个word都embedding成长度为16的向量
batch_size = 128
model = keras.models.Sequential([
    #1.定义一个矩阵大小为:(vocab_size,embedding_dim)
    #2.每个样本的值[1,2,3,4...]都会去查matrix里去查，把1变成1对应的向量、2变成2对应的向量，每个句子都变成了一个max_length * embedding_dim的矩阵
    #3.最后的数据是batch_size * max_length * embedding_dim 的三维矩阵
    keras.layers.Embedding(vocab_size,embedding_dim,input_length = max_length),#第一层是embedding层
    #合并 对于输入batch_size * max_length * embedding_dim 变成 batch_size * embedding_dim的矩阵，把max_length的维度消除
    keras.layers.GlobalAveragePooling1D(),
    #设置全连接层
    keras.layers.Dense(64,activation = "relu"),
    keras.layers.Dense(1,activation="sigmoid")#输出
])

model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

#开始训练,加上batch信息，从训练集里划分出20%作为验证集
history = model.fit(train_data,
                    train_labels,
                    epochs = 30,
                    batch_size = batch_size,
                    validation_split = 0.2
                    )
def plot_learning_curves(history,label,epochs,min_value,max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize = (8,5))
    plt.grid(True)
    plt.axis([0,epochs,min_value,max_value])
    plt.show(block = True)

plot_learning_curves(history, 'accuracy', 30, 0, 1)
plot_learning_curves(history, 'loss', 30, 0, 1)

#在测试集上验证一下
model.evaluate(test_data, test_labels,batch_size = batch_size,)
