# some example code to run tensorflow in limited capacity
import tensorflow as tf

# define the model
a = tf.random_normal([1, 10000])
b = tf.random_normal([10000, 10000])
c = tf.matmul(a, b)
c = tf.matmul(c, b) * 1e-6
c = tf.matmul(c, b)
c = tf.matmul(c, b) * 1e-6
c = tf.matmul(c, b)
c = tf.matmul(c, b) * 1e-6
c = tf.matmul(c, b)
c = tf.matmul(c, b) * 1e-6
c = tf.matmul(c, b)
c = tf.matmul(c, b) * 1e-6
c = tf.matmul(c, b)

# make a config that limits thread use
cfg = tf.ConfigProto()
cfg.intra_op_parallelism_threads = 1
cfg.inter_op_parallelism_threads = 1
# make the session with the config
sess = tf.Session(config=cfg)

# run the model
tf.global_variables_initializer()
sess.run(c)
print(c)
