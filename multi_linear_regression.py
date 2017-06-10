import tensorflow as tf
import numpy as np


x_train = np.random.random((50,2))
y_train = np.random.random((50,1))

X = tf.placeholder(dtype=tf.float32, shape=(50,2))
Y = tf.placeholder(dtype=tf.float32, shape=(50,1))

W = tf.Variable(np.zeros((2,1)), dtype=tf.float32)
b = tf.Variable(np.zeros((1)), dtype=tf.float32)

# y = wx + b
linear_model = tf.add(tf.matmul(X,W),b)
loss = tf.reduce_sum(tf.square(linear_model - Y)) / (2*50)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(1000):
        loss_value,_ =sess.run([loss, train], feed_dict={X:x_train, Y:y_train})
        print loss_value