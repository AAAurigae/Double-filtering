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
#数据集，电影的评论，被分为positive and
imdb = keras.datasets.imdb
word_index = imdb.get_word_index()
vocab_size = 250
index_from = 3
#载入数据
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words = vocab_size,index_from = index_from)
#num_words来设置数据中资料的个数 , index_from 控制是词表中的数据从几开始算
print(train_data[0],train_labels[0])#[1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4,
print(train_data.shape,train_labels.shape)#(25000,) (25000,)
print(len(train_data[0]),len(train_data[1]))#218 189
print(test_data.shape,test_labels.shape)#(25000,) (25000,)


word_index_1 = {k:(v+3) for k, v in word_index.items()}#把所有的id往上偏移 3
#几个特殊字符
word_index['<PAD>'] = 0
word_index['START'] = 1
word_index['UNK'] = 2
word_index['END'] = 3

reverse_word_index = dict([(value, key) for key, value in word_index.items()])

#解码看一下文本是什么：构建词表索引
def decode_review(text_ids):
    return ''.join([reverse_word_index.get(word_id,"UNK") for word_id in text_ids])

decode_review(train_data[0])
'''
"<START> this film was just brilliant casting location scenery story direction everyone's really suited the part 
they played and you could just imagine being there robert <UNK> is an amazing actor and now the same being director 
<UNK> father came from the same scottish island as myself so i loved the fact there was a real connection with this film 
the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was 
released for <UNK> and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end 
it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also <UNK> 
to the two little boy's that played the <UNK> of norman and paul they were just brilliant children are often left out of
the <UNK> list i think because the stars that play them all grown up are such a big profile for the whole film but 
these children are amazing and should be praised for what they have done don't you think the whole story was so lovely 
because it was true and was someone's life after all that was shared with us all"
'''



text_len_li = list(map(len, train_data))
#对数据进行补全（padding）
max_length = 500 #句子长度低于500的句子会被补全，高于500 的会被截断
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,#list of list
    value = word_index['<PAD>'],
    padding = 'post',#post是把padding安在句子的后面，pre是把padding安在句子的前面
    maxlen = max_length
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,#list of list
    value = word_index['<PAD>'],
    padding = 'post',#post是把padding安在句子的后面，pre是把padding安在句子的前面
    maxlen = max_length
)





#打印第一个样本
print(train_data[0])
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
'''
合并+padding的缺点：
    1.信息丢失（多个embedding合并，pad噪音，无主次（有些情感词影响大，但是没体现出来，有些连接词、主语等影响小，造成信息被稀释））
    2.无效计算太多，低效（太多的padding）
'''
