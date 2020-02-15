# coding:utf-8
#  数据类型
import tensorflow as tf     # 引入模块
a = tf.constant([1.0, 2.0])  # 定义一个张量等于 [1.0,2.0]
b = tf.constant([3.0, 4.0])  # 定义一个张量等于 [3.0,4.0]
result = a+b                 # 实现 a加 b的加法
print(result)                # 
print(a)


