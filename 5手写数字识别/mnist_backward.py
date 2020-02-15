# 去掉警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 描述网络参数的训练方法
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward

BATCH_SIZE = 200             # 每轮喂入的图片数量
LEARNING_RATE_BASE = 0.1     # 学习率
LEARNING_RATE_DECAY = 0.99   # 衰减率
REGULARIZER = 0.0001         # 正则化系数
STEPS = 50000                #共训练多少轮
MOVING_AVERAGE_DECAY = 0.99   # 滑动平均衰减率
MODEL_SAVE_PATH = "./model/" # 模型的保存路径
MODEL_NAME = "mnist_model"   #  模型保存名


def backward(mnist):    # 读入数据
    # 占位
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE]) # 真实值
    # 计算
    y = mnist_forward.forward(x, REGULARIZER) # 预测值
    #
    global_step = tf.Variable(0, trainable=False)   # 不可训练
    
    # 交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    
    # tf.reduce_mean(x,axis)函数表示求取矩阵或张量指定维度的平均值。  
    # 这里没有指定第二个参数，所以所有元素取平均值
     # 损失函数
    cem = tf.reduce_mean(ce)  
    # tf.get_collection(“”)函数表示从 collection 集合中取出全部变量生成一个列表。
    # tf.add( )函数表示将参数列表中对应元素相加
    # 总损失值为预测结果与标准答案的损失值加上正则化项
    loss = cem + tf.add_n(tf.get_collection('losses'))
    
    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,   # 学习率
        global_step,         
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,   # 衰减率
        staircase=True)

     # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step) # 滑动平均基数
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    # 实例化
    saver = tf.train.Saver()

    # 初始化所有变量，迭代训练
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        for i in range(STEPS):
             # mnist.train.next_batch将数据输入神经网络
             # 表示一次将200个样本的像素值和标签赋值给xs和ys, 故xs的形状为（200，784）
             # ys的形状为（200，10）  # 10是标签
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # 保存模型
                #  保存当前图结构的.meta 文件、 保存当前参数名的.index 文件、 保存当前参数的.data 文件
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    print("train data size:", mnist.train.num_examples)# 训练集数量 55000
    print("validation data size:%d",mnist.validation.num_examples)# 验证集5000
    print("test data size:%d",mnist.test.num_examples) # 测试集10000
    backward(mnist)
    
if __name__ == '__main__':
    main()
