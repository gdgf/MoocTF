# coding:utf-8
# 去掉警告
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#  会话
import tensorflow as tf  # 引入模块

x = tf.constant([[1.0, 2.0]])  # 定义一个2阶张量等于[[1.0,2.0]]
w = tf.constant([[3.0], [4.0]])  # 定义一个2阶张量等于[[3.0],[4.0]]
y = tf.matmul(x, w)  # 实现xw矩阵乘法
# 生成权重随机数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
print(y)  # 打印出结果
with tf.Session() as sess:
    print(sess.run(y))  # 执行会话并打印出执行后的结果
    sess.run(tf.global_variables_initializer())  # 初始化所有带优化的参数
    print(w1)
    print(sess.run(w1))
