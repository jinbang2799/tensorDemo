import tensorflow as tf

# Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input3, input2)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    print(sess.run([add, mul]))

# Feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
mul = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(mul, feed_dict={input1: [7.], input2: [3.]}))
