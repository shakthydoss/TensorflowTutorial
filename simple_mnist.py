import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    loss_value_arr = []
    for _ in range(10):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      loss_value, _ = sess.run([cross_entropy,train_step], feed_dict={x: batch_xs, y_: batch_ys})
      print loss_value
      loss_value_arr.append(loss_value)
    plt.scatter(range(10),loss_value_arr)
    plt.show()



