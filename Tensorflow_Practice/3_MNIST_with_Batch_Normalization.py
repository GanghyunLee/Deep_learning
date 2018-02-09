import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 784], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')
phase = tf.placeholder(tf.bool, name='phase') # to use Batch Normalization only in Training time!!

# 784 -> 256(1st layer) -> 256(2nd layer) -> 10(result 0~9)
W1 = tf.Variable(tf.random_normal([784, 256], stddev = 0.01), name='W1')
b1 = tf.Variable(tf.zeros(shape = [1, 256], dtype=tf.float32), name='b1')

L1 = tf.matmul(X, W1) + b1
L1 = tf.contrib.layers.batch_norm(L1, center=True, scale=True, is_training=phase)
L1 = tf.nn.relu(L1, name='Layer1')

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01), name='W2')
b2 = tf.Variable(tf.zeros(shape = [1, 256], dtype=tf.float32), name='b2')

L2 = tf.matmul(L1, W2) + b2
L2 = tf.contrib.layers.batch_norm(L2, center=True, scale=True, is_training=phase)
L2 = tf.nn.relu(L2, name='Layer2')

W3 = tf.Variable(tf.random_normal([256, 10], stddev = 0.01), name='W3')
b3 = tf.Variable(tf.zeros(shape = [1, 10], dtype=tf.float32), name='b3')
model = tf.matmul(L2, W3) + b3

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # moving_mean, moving_variance

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
with tf.control_dependencies(update_ops):
        # ★ Ensures that we execute the update_ops before performing the train_step
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 미니배치 계산
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(6):
        total_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, phase:True})
            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

    is_correct = tf.equal(tf.arg_max(model, 1), tf.arg_max(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print("학습 정확도 : ", sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels, phase: True}))
    print("테스트 정확도 : ",  sess.run(accuracy, feed_dict = {X: mnist.test.images, Y: mnist.test.labels, phase:False}))