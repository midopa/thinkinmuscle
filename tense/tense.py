import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

learning_rate = 1e-4
epochs = 1
iterations = 5000
batch_size = 50


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


# reshape input image into an actual image
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])

# layer 1: convolution and max pooling
# 5x5 patches, output 32 features
W_1 = make_weights([5, 5, 1, 32])
b_1 = make_bias([32])
y_1 = max_pool(
    tf.nn.relu(conv(x_img, W_1) + b_1)
)

# layer 2: another convolution and max pooling
W_2 = make_weights([5, 5, 32, 64])
b_2 = make_bias([64])
y_2 = max_pool(
    tf.nn.relu(conv(y_1, W_2) + b_2)
)

# layer 3: dense network =
W_3 = make_weights([7 * 7 * 64, 1024])
b_3 = make_bias([1024])
# reshape the image outputs into a vector
y_2_vec = tf.reshape(y_2, [-1, 7 * 7 * 64])
y_3 = tf.nn.relu(tf.matmul(y_2_vec, W_3) + b_3)

# output layer
W_o = make_weights([1024, 10])
b_o = make_bias([10])
y_o = tf.matmul(y_3, W_o) + b_o

# actual output, labels
a = tf.placeholder(tf.float32, [None, 10])

# training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_o, a))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# debugging
correct_prediction = tf.equal(tf.argmax(y_o, 1), tf.argmax(a, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# with tf.InteractiveSession() as sess:
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    def check_accuracy(step, input, actual):
        train_accuracy = sess.run(accuracy, feed_dict={x: input, a: actual})
        print('step {}, training accuracy {}'.format(step, train_accuracy))

    for _ in range(epochs):
        for i in range(iterations):
            batch_x, batch_a = mnist.train.next_batch(batch_size)

            if i % 100 == 0:
                check_accuracy(i, batch_x, batch_a)

            sess.run(train_step, feed_dict={x: batch_x, a: batch_a})

        check_accuracy('complete', mnist.test.images, mnist.test.labels)
