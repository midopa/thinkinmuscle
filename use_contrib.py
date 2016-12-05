#!/usr/bin/env python
import logging

import coloredlogs
import matplotlib
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

coloredlogs.install(
    level='debug',
    datefmt='%H:%M:%S',
    isatty=True
)
logging.getLogger(name='tensorflow').setLevel(logging.ERROR)
log = logging.getLogger(name='derp')

matplotlib.use('tkagg')
# important for this to be after setting the backend
from matplotlib import pyplot
pyplot.ion()

mnist = input_data.read_data_sets('mnist_data')

# hyper params
drop_out_prob = 0.0
network_shape = [128, 64, 32]
epochs = 20
batch_size = 50

# plotting data
plot_samples = [0]
plot_train_acc = [0]
plot_val_acc = [0]
plot_epochs = [0]
plot_test_acc = [0]

net = tf.contrib.learn.DNNClassifier(
    feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(mnist.train.images),
    hidden_units=network_shape,
    n_classes=10,
    dropout=drop_out_prob
)


def plot_accs():
    pyplot.cla()
    pyplot.plot(plot_samples, plot_train_acc, 'b--+', label='training acc')
    pyplot.plot(plot_samples, plot_val_acc, 'r--+', label='val acc')
    pyplot.plot(plot_epochs, plot_test_acc, 'g-+', label='test acc')
    pyplot.legend(loc='lower right')
    pyplot.pause(1e-3)


for e in range(1, epochs+1):
    log.debug('epoch {}'.format(e))

    net.fit(
        mnist.train.images,
        mnist.train.labels.astype('int'),
        steps=int(mnist.train.num_examples/batch_size),
        batch_size=batch_size
    )

    # num_samples = (e-1) * train_size + (i+1) * acc_check_steps * batch_size
    # plot_samples.append(num_samples)
    # x, a = mnist.train.next_batch(batch_size)
    # train_acc = net.evaluate(x, a.astype('int'))['accuracy']
    # plot_train_acc.append(train_acc)
    #
    # x, a = mnist.validation.next_batch(batch_size)
    # val_acc = net.evaluate(x, a.astype('int'))['accuracy']
    # plot_val_acc.append(val_acc)
    #
    # log.debug('samples {}, train acc {:0.3f}, val acc {:0.3f}'.format(num_samples, train_acc, val_acc))
    #
    # plot_accs()

    test_acc = net.evaluate(
        mnist.test.images,
        mnist.test.labels.astype('int')
    )['accuracy']
    log.debug('test acc: {:0.3f}'.format(test_acc))
    plot_epochs.append(e * mnist.train.num_examples)
    plot_test_acc.append(test_acc)
    plot_accs()
