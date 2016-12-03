#!/usr/bin/env python
import logging
import signal
import sys

import coloredlogs
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tense.tense import ThinkinMuscle

coloredlogs.install(
    level='debug',
    fmt='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger(name='tensorflow').setLevel(logging.INFO)
log = logging.getLogger(name='derp')


def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets('mnist_data', one_hot=True)

learning_rate = 1e-4
epochs = 1
iterations = 20000
batch_size = 50

network = ThinkinMuscle()

# reshape input image into an actual image
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])

# simple, layered, fully connected neural net
# network.add_layer(
#     [784, 1024],
#     lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b),
#     input_override=x
# )
# network.add_layer(
#     [1024, 512],
#     lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b)
# )
# network.add_layer(
#     [512, 100],
#     lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b)
# )
# network.add_layer(
#     [100, 10],
#     lambda y, W, b: tf.matmul(y, W) + b
# )

# convolution and max pooling
# 5x5 patches, output 32 features
network.add_layer(
    [5, 5, 1, 32],
    lambda y, W, b: max_pool(tf.nn.relu(conv(y, W) + b)),
    input_override=x_img
)
# convolution and max pooling
network.add_layer(
    [5, 5, 32, 64],
    lambda y, W, b: max_pool(tf.nn.relu(conv(y, W) + b))
)
# dense network, flatten convolution outputs
# reshape the image outputs into a vector
network.add_layer(
    [7 * 7 * 64, 1024],
    lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b),
    input_override=tf.reshape(network.ys[-1], [-1, 7 * 7 * 64])
)
# output layer
_, __, y_out = network.add_layer(
    [1024, 10],
    lambda y, W, b: tf.matmul(y, W) + b
)

# actual output, labels
a = tf.placeholder(tf.float32, [None, 10])

# training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, a))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# debugging
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(a, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def check_test():
    test_acc = accuracy.eval(feed_dict={x: mnist.test.images, a: mnist.test.labels})
    log.info('done, test acc {:0.3f}'.format(test_acc))


def check_accuracy(step, in_x, actual):
    train_acc = accuracy.eval(feed_dict={x: in_x, a: actual})
    in_x, actual = mnist.validation.next_batch(batch_size)
    val_acc = accuracy.eval(feed_dict={x: in_x, a: actual})
    log.debug('step {}, train acc {:0.3f}, val acc {:0.3f}'.format(step, train_acc, val_acc))


def signal_handler(sig_num, frame):
    """
    executes if the program is interrupted (ctrl+c) and does a check against the
    test set.
    """
    log.info('aborted')
    check_test()
    sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(iterations):
    batch_x, batch_a = mnist.train.next_batch(batch_size)

    if i % 100 == 0:
        check_accuracy(i, batch_x, batch_a)

    sess.run(train_step, feed_dict={x: batch_x, a: batch_a})

check_test()
