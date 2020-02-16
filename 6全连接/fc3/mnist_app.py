# coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 实现对输入图片的收写数字识别
# 应用程序
import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward

def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        # x站位
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 加载ckpt
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 存在cpkt,则把参数复原
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 执行预测操作
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


# 预处理
def pre_pic(picName):
    img = Image.open(picName)
    reIm = img.resize((28, 28), Image.ANTIALIAS)  # 整理图片的大小、消除锯齿的方法resize
    im_arr = np.array(reIm.convert('L'))          # 变成灰度图，然后变成矩阵
    threshold = 50
    # 输入图片反色，二值化处理
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    # 整理形状，1行784列
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)

    img_ready = np.multiply(nm_arr, 1.0 / 255.0)
    return img_ready   # 整理好的待识别图片

def application():
    # 输入识别的图片，给出路径
    testNum = input("input the number of test pictures:")
    for i in range(int(testNum)):
        testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic) # 预处理
        preValue = restore_model(testPicArr) # 图片喂入神经网络
        print("The prediction number is:", preValue)

def main():
    application()


if __name__ == '__main__':
    main()
