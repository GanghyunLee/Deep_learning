import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')
keep_prob = tf.placeholder(tf.float32)

# 784 -> 256(1st layer) -> 256(2nd layer) -> 10(result 0~9)
W1 = tf.Variable(tf.random_normal([784, 256], stddev = 0.01), name='W1')
b1 = tf.Variable(tf.zeros(shape = [1, 256], dtype=tf.float32), name='b1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1, name='Layer1')
L1 = tf.nn.dropout(L1, keep_prob)   # Dropout

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01), name='W2')
b2 = tf.Variable(tf.zeros(shape = [1, 256], dtype=tf.float32), name='b2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2, name='Layer2')
L2 = tf.nn.dropout(L1, keep_prob)   # Dropout

W3 = tf.Variable(tf.random_normal([256, 10], stddev = 0.01), name='W3')
b3 = tf.Variable(tf.zeros(shape = [1, 10], dtype=tf.float32), name='b3')
model = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 미니배치 계산
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(30):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8}) # Train Dropout = 0.8
            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

    is_correct = tf.equal(tf.arg_max(model, 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print("Test 정확도 : ",  sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})) # ★ Test Dropout = 1

