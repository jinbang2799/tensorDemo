# 导入依赖库
# 这是Python的一种开源的数值计算扩展，非常强大
import numpy as np
# 导入tensorflow ##构造数据##
import tensorflow as tf

# 随机生成100个类型为float32的值
x_data = np.random.rand(100).astype(np.float32)
# 定义方程式y = x_data * A + B
y_data = x_data * 0.1 + 0.3
##-------## ##建立TensorFlow神经计算结构##
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = weight * x_data + biases
##-------##
# 判断与正确值的差距
loss = tf.reduce_mean(tf.square(y - y_data))
# 根据差距进行反向传播修正参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 建立训练器
train = optimizer.minimize(loss)
# 初始化TensorFlow训练结构
init = tf.initialize_all_variables()
# 建立TensorFlow训练会话
sess = tf.Session()
sess.run(init)
# 将训练结构装载到会话中
# 循环训练400次
for step in range(400):
    # 使用训练器根据训练结构进行训练
    sess.run(train)
    # 每20次打印一次训练结果
    if step % 20 == 0:
        # 训练次数，A值，B值
        print(step, sess.run(weight), sess.run(biases))
