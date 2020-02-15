# coding:utf-8
# 去掉警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 多组喂养
"""
   第一组喂体积0.7、重量0.5，第二组喂体积0.2、重量0.3，
   第三组喂体积0.3 、重量0.4，第四组喂体积0.4、重量 0.5. 
"""
import tensorflow as tf
# 定义输入和参数
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
with tf.Session() as sess:
    # 变量初始化/赋初值
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y=: \n", sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
