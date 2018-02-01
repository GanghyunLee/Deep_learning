import numpy as np
import tensorflow as tf

#=================================================
X_train = np.array([
    [1.,0.,1.,0.],
    [1.,1.,0.,0.],
])

Y_train = np.array([[0.,1.,1.,0.]])
#=================================================

X = tf.placeholder(dtype = tf.float32, shape = [2, None], name = "X")
Y = tf.placeholder(dtype = tf.float32, shape = [1, None], name = "Y")

W1 = tf.get_variable(name="W1", shape=[10, 2], initializer = tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable(name="W2", shape=[1, 10], initializer = tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros(shape = [10, 1]), name="b1")
b2 = tf.Variable(tf.zeros(shape = [1, 1]), name="b2")

z1 = tf.matmul(W1, X) + b1
a1 = tf.nn.relu(z1)
z2 = tf.matmul(W2, a1) + b2
a2 = tf.nn.sigmoid(z2)

cost = -tf.reduce_mean( (Y * tf.log(a2)) + ((1 - Y) * tf.log(1 - a2)) )
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predicted = tf.cast(a2 > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(5001):
        sess.run(optimizer, feed_dict = {X: X_train, Y: Y_train})

    # Predict
    p, a = sess.run([predicted, accuracy], feed_dict = {X: X_train, Y: Y_train})

    print("Predicted : " + str(p))
    print("Accuracy : " + str(a))

