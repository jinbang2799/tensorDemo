import tensorflow as tf
import numpy as np

# 使用numpy生成100个随机点
x_data = np.random.rand(1000000)
y_data = x_data * 0.1 + 0.2

# 构造一个线性模型
k = tf.Variable(0.)
b = tf.Variable(0.)
y = k * x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
# 构造一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.6)
# 最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10001):
        sess.run(train)
        if (i % 1000 == 0):
            print(i, sess.run([k, b]))
