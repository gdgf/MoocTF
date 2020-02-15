# co ding:utf-8
# 去掉警告
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#  前向传播，单组喂养
# 第一组喂体积 0.7 、重量 0.5
# 两层简单神经网络（全连接）
import tensorflow as tf

# 定义输入和参数
x = tf.placeholder(tf.float32, shape=(1, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 用会话计算结果
with tf.Session() as sess:
     # 要初始化的所有数据定义为init_op
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("y=\n", sess.run(y, feed_dict={x: [[0.7, 0.5]]}))