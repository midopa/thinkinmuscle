import signal
import sys
import time
from typing import List, Callable, Tuple

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

learning_rate = 1e-4
epochs = 1
iterations = 20000
batch_size = 50


def make_weights(shape: List[int]) -> tf.Variable:
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def make_bias(shape: List[int]) -> tf.Variable:
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# layers of weights, biases, and outputs
Ws = []  # type: List[tf.Variable]
bs = []  # type: List[tf.Variable]
ys = []  # type: List[tf.Tensor]


def add_layer(
        weights_shape: List[int],
        activation: Callable[[tf.Tensor, tf.Variable, tf.Variable], tf.Tensor],
        input_override: tf.Variable = None
) -> Tuple[tf.Variable, tf.Variable, tf.Tensor]:
    """
    adds a layer to the neural net.
    TODO check compatibility with the previous layer's shape.

    :param weights_shape: shape of the weights. the last element is used as the
        shape of the bias vector
    :param activation: function that will generate/provide a tensor op that will
        be used as the activation function for the nodes in this layer. it gets
        provided the input tensor and previous weights and biases
    :param input_override: if given, will be passed as input to the activation
        function, instead of the previous layer's output.
    :return: latest weights, bias, and output
    """
    Ws.append(make_weights(weights_shape))
    bs.append(make_bias([weights_shape[-1]]))

    input_tensor = ys[-1] if input_override is None else input_override
    ys.append(activation(input_tensor, Ws[-1], bs[-1]))

    activation(input_tensor, Ws[-1], bs[-1])

    return Ws[-1], bs[-1], ys[-1]


# reshape input image into an actual image
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])

# simple, layered, fully connected neural net
# add_layer(
#     [784, 1024],
#     lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b),
#     input_override=x
# )
# add_layer(
#     [1024, 512],
#     lambda y, W, b: tf.nn.relu(tf.matmul(y, W) + b)
# )
# add_layer(
#     [512, 100],
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


def check_test():
    test_acc = accuracy.eval(feed_dict={x: mnist.test.images, a: mnist.test.labels})
    t_now = time.time() - t_start
    print('{:0.1f}s: done, test acc {:0.3f}'.format(t_now, test_acc))


def check_accuracy(step, in_x, actual):
    train_acc = accuracy.eval(feed_dict={x: in_x, a: actual})
    in_x, actual = mnist.validation.next_batch(batch_size)
    val_acc = accuracy.eval(feed_dict={x: in_x, a: actual})
    t_now = time.time() - t_start
    print('{:0.1f}s: step {}, train acc {:0.3f}, val acc {:0.3f}'.format(t_now, step, train_acc, val_acc))


def signal_handler(sig_num, frame):
    t_now = time.time() - t_start
    print('{:0.1f}s: aborted'.format(t_now))
    check_test()
    sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)


for i in range(iterations):
    batch_x, batch_a = mnist.train.next_batch(batch_size)

    if i % 100 == 0:
        check_accuracy(i, batch_x, batch_a)

    sess.run(train_step, feed_dict={x: batch_x, a: batch_a})

check_test()
