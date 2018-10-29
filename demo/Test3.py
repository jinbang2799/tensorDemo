import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 使用numpy生成200个随机点
x_data = np.linspace(-1000, 1000, 4000)[:, np.newaxis]
noise = np.random.normal(0, 0.0001, x_data.shape)
y_data = x_data * 5 + noise

x = tf.placeholder(tf.float64)
y = tf.placeholder(tf.float64)

W = tf.Variable([5], dtype=tf.float64)
b = tf.Variable([0], dtype=tf.float64)

linear_model = W * x + b

loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.AdamOptimizer(0.5)
# optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train, {x: x_data, y: y_data})
        if i % 10 == 0:
            print("Epoch:", '%04d' % i, "cost=", "W=", sess.run(W), "b=", sess.run(b))
    print(sess.run([W, b]))

    # 获得预测值
    prediction_value = sess.run(linear_model, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
