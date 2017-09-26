import numpy as np
import tensorflow as tf


def lrelu(x, leak=0.2):

    return tf.maximum(x, leak * x)


def linear(x, n_out, bias=True, name='linear'):

    with tf.variable_scope(name):

        n_in = x.shape[-1]

        # Initialize w
        w_init_std = np.sqrt(1.0 / n_out)
        w_init = tf.truncated_normal_initializer(0.0, w_init_std)
        w = tf.get_variable('w', shape=[n_in, n_out], initializer=w_init)

        # Dense mutliplication
        x = tf.matmul(x, w)

        if bias:

            # Initialize b
            b_init = tf.constant_initializer(0.0)
            b = tf.get_variable('b', shape=(n_out,), initializer=b_init)

            # Add b
            x = x + b

        return x


def conv2d(x, n_in, n_out, k, s, p, bias=True, name='conv2d', stddev=0.02):

    with tf.variable_scope(name):

        strides = [1, s, s, 1]

        # Initialize weigth
        w_init = tf.truncated_normal_initializer(stddev=stddev)
        w = tf.get_variable('w', [k, k, n_in, n_out], initializer=w_init)

        # Compute conv
        conv = tf.nn.conv2d(x, w, strides=strides, padding=p)

        if bias:
            # Initialize bias
            b_init = tf.constant_initializer(0.0)
            b = tf.get_variable('b', shape=(n_out,), initializer=b_init)

            # Add bias
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv
