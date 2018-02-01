import numpy as np
import tensorflow as tf

#=================================================
X_train = np.array([
    [1.,0.,1.,0.],
    [1.,1.,0.,0.],
])

Y_train = np.array([[0.,1.,1.,0.]])
#=================================================

X = tf.placeholder(dtype = tf.float32, shape=[2, None], name="X")
Y = tf.placeholder(dtype = tf.float32, shape=[1, None], name="Y")

W1 = tf.get_variable("W1", shape = [10, 2], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape = [1, 10], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([10, 1]), name = "b1")
b2 = tf.Variable(tf.zeros([1, 1]), name = "b2")

z1 = tf.matmul(W1, X) + b1
a1 = tf.nn.relu(z1)

z2 = tf.matmul(W2, a1) + b2
a2 = tf.nn.sigmoid(z2)

cost = tf.reduce_mean(tf.square(a2 - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(5001):
        sess.run(optimizer, feed_dict = {X: X_train, Y: Y_train})

    print(sess.run(a2, feed_dict = {X: X_train, Y: Y_train}))

