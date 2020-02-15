# coding:utf-8
#  计算图,只搭建网络，不计算
import tensorflow as tf           # 引入模块
x = tf.constant([[1.0, 2.0]])     # 定义一个2阶张量等于[[1.0,2.0]]
w = tf.constant([[3.0], [4.0]])   # 定义一个2阶张量等[[3.0],[4.0]]
y = tf.matmul(x, w)               # 实现xw矩阵乘法
print(y)                          # 打印出结果
