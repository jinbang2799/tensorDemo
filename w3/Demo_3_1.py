# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# In[6]:

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200, dtype='float32')[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])

k = tf.Variable(0.)
k2 = tf.Variable(0.)
b = tf.Variable(0.)
y = k * tf.square(x_data) + k2 * x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(y, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()

# In[ ]:
