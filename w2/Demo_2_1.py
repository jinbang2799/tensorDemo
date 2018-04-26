import tensorflow as tf

# 创建一个常量op
m1 = tf.constant([[3, 3]])
# 创建一个常量op
m2 = tf.constant([[2], [3]])
# 创建一个矩阵乘法op，传入m1,m2
product = tf.matmul(m1, m2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
