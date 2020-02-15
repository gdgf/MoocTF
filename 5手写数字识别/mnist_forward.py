# 去掉警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# 描述网络结构

import tensorflow as tf
INPUT_NODE = 784    # 一张图片的像素值
OUTPUT_NODE = 10    # 输出10个数，表示输出的每个数，表示10分类
LAYER1_NODE = 500   # d定义隐藏层的节点个数

def get_weight(shape, regularizer):
    # 随机生成w
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    # 如果使用正则化，则将每个变量的正则化损失加入到总的损失中
    if regularizer != None:tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    #
    return w

# 定义偏置
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

# 搭建网络
def forward(x, regularizer):

    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)  # 参数
    b1 = get_bias([LAYER1_NODE])  # 偏置
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1) # 结果

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])   # 偏置
    y = tf.matmul(y1, w2) + b2
    return y
