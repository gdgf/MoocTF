import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant("Hello world")
sess = tf.Session()
print(sess.run(hello))

t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
print(tf.shape(t)) # [2, 2, 3]
