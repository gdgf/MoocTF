# coding:utf-8
# 去掉警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 测试浮现了


import time  # 为了延迟
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

TEST_INTERVAL_SECS = 5 # 程序循环的额时间


def test(mnist):
    with tf.Graph().as_default() as g:  # 复现计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None) # 计算出y的值

        # 实例化带滑动平均的saver对象
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 如果有的haunted恢复参数
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    # 没有模型和参数，
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
