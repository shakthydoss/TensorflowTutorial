import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# In this program we will playaround with
# simple linear regression that is y = wx + b model in tensor flow.
# Here input and output is single dimension.


# Generating training data
train_X = np.linspace(-1, 1, 50)
train_Y = 2 * train_X + np.random.random(*train_X.shape) * 0.3

# Generating placeholder for input and output.
# these placeholder will be used in tensor flow graph
# to pass actual inputs and outputs while executing the program.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# tensor flow variables
W = tf.Variable(np.random.random(), name="W")
b = tf.Variable(np.random.random(), name="b")

# model y = wx+ b
linear_model = tf.add(tf.multiply(X, W), b)

# loss function.
loss = tf.reduce_mean(tf.square(linear_model - Y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    loss_val_arr = []
    for _ in range(100):
        loss_val, _ = sess.run([loss, train], feed_dict={X: train_X, Y: train_Y})
    print("W=", sess.run(W), "b=", sess.run(b), '\n')
