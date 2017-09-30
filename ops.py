import numpy as np
import tensorflow as tf


def lrelu(x, leak=0.2):

    return tf.maximum(x, leak * x)


def linear(x, n_out, name='linear'):

    with tf.variable_scope(name):

        n_in = x.shape[-1]

        w_init = tf.truncated_normal_initializer(0.0, np.sqrt(1.0 / n_out))
        w = tf.get_variable('w', shape=[n_in, n_out], initializer=w_init)

        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable('b', shape=(n_out,), initializer=b_init)

        x = tf.matmul(x, w) + b

        return x


def conv2d(x, n_out, k, s, p, stddev=0.02, name='conv2d'):

    with tf.variable_scope(name):

        n_in = x.shape[-1]
        strides = [1, s, s, 1]

        w_init = tf.truncated_normal_initializer(stddev=stddev)
        w = tf.get_variable('w', [k, k, n_in, n_out], initializer=w_init)

        conv = tf.nn.conv2d(x, w, strides=strides, padding=p)

        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable('b', shape=(n_out,), initializer=b_init)

        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv


def maxpool2d(x, k, s, p, name='maxpool2d'):
    strides = [1, s, s, 1]
    ksize = [1, k, k, 1]

    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=p, name=name)


def fc(x, n_out, name='fc'):

    with tf.variable_scope(name):

        shape = x.shape.as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])

        w_init = tf.truncated_normal_initializer(0.0, np.sqrt(1.0 / n_out))
        w = tf.get_variable('w', [x.shape[-1], n_out], initializer=w_init)

        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable('b', [n_out], initializer=b_init)

        return tf.matmul(x, w) + b
