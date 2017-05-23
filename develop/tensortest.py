import tensorflow as tf

with tf.variable_scope('my_scope'):
    v = tf.get_variable("E", [40, 1, 1, 128], tf.float32, tf.random_normal_initializer(stddev=1))



test = tf.placeholder(tf.int64, shape=None, name='test')
print (test)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (v.eval())
