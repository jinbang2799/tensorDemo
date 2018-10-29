import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.placeholder(tf.float64)
y = tf.placeholder(tf.float64)

W = tf.Variable([0.3], dtype=tf.float64)
b = tf.Variable([-0.3], dtype=tf.float64)

linear_model = W * x + b

loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3.2, 4], y: [0, -1.01, -2, -3]})
    print(sess.run([W, b]))
