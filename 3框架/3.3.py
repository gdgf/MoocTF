# coding:utf-8
# 去掉警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0导入模块，生成模拟数据集。
import tensorflow as tf
import numpy as np  # 科学计算模块

BATCH_SIZE = 8       # 一次喂入神经网络多少组数据
SEED = 23455         # 保持结果一样

# 基于seed产生随机数
rdm = np.random.RandomState(SEED)
# 随机数返回32行2列的矩阵，表示32组特征（体积和重量），作为输入数据集
X = rdm.rand(32, 2)
# 从X这个32行2列的矩阵中，取出一行，判断如果和小于1 给Y赋值1， 如果和不小于1， Y赋值0
# 作为输入数据集的标签（正确答案）(这里是虚拟生成的数据)(体积加重量小于1)
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print("X:\n", X)
print("Y_:\n", Y_)

# 1定义神经网络的输入、参数和输出,定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 参数刚开始随机生成
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 前向传播，矩阵相乘
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


# 2定义损失函数及反向传播方法。
# 均方误差
loss_mse = tf.reduce_mean(tf.square(y - y_))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
# train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

# 3生成会话，训练STEPS轮
with tf.Session() as sess:
    # 初始化变量，输出目前（未经训练）的参数取值。
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("训练前的参数:")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")

    # 训练模型。
    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        #print([],y)
        # 每500次输出一次total_loss
        if i % 500 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))
         
    # 输出训练后的参数取值。
    print("训练后的参数：")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
