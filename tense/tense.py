import signal
import time

import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

learning_rate = 1e-4
epochs = 1
iterations = 20000
batch_size = 500
val_batch_size = 500


def make_weights(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def make_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# layers of weights, biases, and outputs
Ws = []
bs = []
ys = []


def add_layer(weights_shape, activation, input_override=None):
    Ws.append(make_weights(weights_shape))
    bs.append(make_bias([weights_shape[-1]]))

    input_tensor = ys[-1] if input_override is None else input_override
    ys.append(activation(input_tensor, Ws[-1], bs[-1]))

    return Ws[-1], bs[-1], ys[-1]


# reshape input image into an actual image
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])

# # simple, layered, fully connected neural net
# add_layer(
#     [784, 500],
#     lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b),
#     input_override=x
# )
# add_layer(
#     [500, 300],
#     lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b)
# )
# add_layer(
#     [300, 100],
#     lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b)
# )
# add_layer(
#     [100, 10],
#     lambda y, W, b: tf.matmul(y, W) + b
# )

# convolution and max pooling
# 5x5 patches, output 32 features
add_layer(
    [5, 5, 1, 32],
    lambda y, W, b: max_pool(tf.nn.relu(conv(y, W) + b)),
    input_override=x_img
)

# convolution and max pooling
add_layer(
    [5, 5, 32, 64],
    lambda y, W, b: max_pool(tf.nn.relu(conv(y, W) + b))
)

# dense network, flatten convolution outputs
# reshape the image outputs into a vector
add_layer(
    [7 * 7 * 64, 1024],
    lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b),
    input_override=tf.reshape(ys[-1], [-1, 7 * 7 * 64])
)

# output layer
add_layer(
    [1024, 10],
    lambda y, W, b: tf.matmul(y, W) + b
)

# output vector
y_out = ys[-1]

# actual output, labels
a = tf.placeholder(tf.float32, [None, 10])

# training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_out, a))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# debugging
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(a, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
t_start = time.time()


def check_accuracy(step, input, actual):
    train_acc = accuracy.eval(feed_dict={x: input, a: actual})
    input, actual = mnist.validation.next_batch(val_batch_size)
    val_acc = accuracy.eval(feed_dict={x: input, a: actual})
    t_now = time.time() - t_start
    print('{:0.3f}s: step {}, train acc {:0.3f}, val acc {:0.3f}'.format(t_now, step, train_acc, val_acc))


def signal_handler(sig_num, frame):
    check_accuracy('aborted', mnist.test.images, mnist.test.labels)
    sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)


for i in range(iterations):
    batch_x, batch_a = mnist.train.next_batch(batch_size)

    if i % 33 == 0:
        check_accuracy(i, batch_x, batch_a)

    sess.run(train_step, feed_dict={x: batch_x, a: batch_a})

check_accuracy('complete', mnist.test.images, mnist.test.labels)
