from keras.layers import Input, Activation, Dropout, Flatten, Dense, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

import tensorflow as tf
import ops


class Model(object):

    def __init__(self, name):
        self.name = name

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model


class Generator(Model):

    def __init__(self, img_size=(128, 128), img_dim=1, z_size=100, k_size=5, layer_n=3, smallest_hidden_unit_n=128, name='generator',
                 is_bn=True, batch_size=256):

        super(Generator, self).__init__(name)

        self.img_size = img_size
        self.img_dim = img_dim
        self.z_size = z_size
        self.k_size = k_size
        self.layer_n = layer_n
        self.smallest_hidden_unit_n = smallest_hidden_unit_n
        self.name = name
        self.is_bn = is_bn
        self.batch_size = batch_size

    def __call__(self, x, is_reuse=False):

        with tf.variable_scope(self.name) as scope:

            if is_reuse:
                scope.reuse_variables()

            unit_size = self.img_size[0] // (2 ** self.layer_n)
            unit_n = self.smallest_hidden_unit_n * (2 ** (self.layer_n - 1))

            x = ops.linear(x, unit_size * unit_size * unit_n)
            x = tf.reshape(x, (self.batch_size, unit_size, unit_size, unit_n))
            x = tf.contrib.layers.batch_norm(x, fused=True)
            x = tf.nn.relu(x)

            for i in range(self.layer_n):
                with tf.variable_scope('layer{}'.format(i)):
                    unit_n_prev = unit_n
                    if i == self.layer_n - 1:
                        unit_n = self.img_dim
                    else:
                        unit_n = self.smallest_hidden_unit_n * (2 ** (self.layer_n - i - 2))
                    x_shape = x.get_shape().as_list()
                    new_height = 2 * x_shape[1]
                    new_width = 2 * x_shape[2]
                    x = tf.image.resize_nearest_neighbor(x, (new_height, new_width))
                    x = ops.conv2d(x, unit_n_prev, unit_n, self.k_size, 1, 'SAME')
                    if i != self.layer_n - 1:
                        x = tf.contrib.layers.batch_norm(x, fused=True)
                        x = tf.nn.relu(x)
            x = tf.nn.tanh(x)

            return x


class Discriminator(Model):

    def __init__(self, img_size=(128, 128), img_dim=1, k_size=5, layer_n=3, smallest_hidden_unit_n=128, name='discriminator',
                 is_bn=True, batch_size=256):
        super(Discriminator, self).__init__(name)

        self.img_size = img_size
        self.img_dim = img_dim
        self.k_size = k_size
        self.layer_n = layer_n
        self.smallest_hidden_unit_n = smallest_hidden_unit_n
        self.name = name
        self.is_bn = is_bn
        self.batch_size = batch_size

    def __call__(self, x, is_reuse=False):
        with tf.variable_scope(self.name) as scope:

            if is_reuse:
                scope.reuse_variables()

            unit_n_prev = self.img_dim
            unit_n = self.smallest_hidden_unit_n

            for i in range(self.layer_n):
                with tf.variable_scope('layer{}'.format(i + 1)):
                    x = ops.conv2d(x, unit_n_prev, unit_n, self.k_size, 2, 'SAME')
                    if self.is_bn and i != 0:
                        tf.contrib.layers.batch_norm(x, fused=False, scope='bn')
                    x = ops.lrelu(x)
                    unit_n_prev = unit_n
                    unit_n = self.smallest_hidden_unit_n * (2 ** (i + 1))

            x = tf.reshape(x, (self.batch_size, -1))
            x = ops.linear(x, 1, bias=False)

            return x


def Classifier(img_size=(256, 256), img_dim=1, class_n=26):
    cl_inp = Input(shape=(img_size[0], img_size[0], img_dim))

    cl_1 = Conv2D(96, (8, 8), strides=(4, 4), padding='same')(cl_inp)
    cl_1 = Activation('relu')(cl_1)
    cl_1 = MaxPool2D((3, 3), strides=(2, 2))(cl_1)

    cl_2 = Conv2D(256, (5, 5), padding='same')(cl_1)
    cl_2 = Activation('relu')(cl_2)
    cl_2 = MaxPool2D((3, 3), strides=(2, 2))(cl_2)

    cl_3 = Conv2D(384, (3, 3), padding='same')(cl_2)
    cl_3 = Activation('relu')(cl_3)

    cl_4 = Conv2D(384, (3, 3), padding='same')(cl_3)
    cl_4 = Activation('relu')(cl_4)

    cl_5 = Conv2D(256, (3, 3), padding='same')(cl_4)
    cl_5 = Activation('relu')(cl_5)
    cl_5 = MaxPool2D((3, 3), strides=(2, 2))(cl_5)

    cl_6 = Flatten()(cl_5)
    cl_6 = Dense(4096)(cl_6)
    cl_6 = Dropout(0.5)(cl_6)

    cl_7 = Dense(4096)(cl_6)
    cl_7 = Dropout(0.5)(cl_7)

    cl_8 = Dense(class_n, activation='softmax')(cl_7)

    model = Model(inputs=cl_inp, outputs=cl_8)

    return model


def ClassifierMin(img_size=(64, 64), img_dim=1, class_n=26):
    inp = Input(shape=(img_size[0], img_size[0], img_dim))

    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    x = Dense(2048)(x)
    x = Dropout(0.5)(x)

    x = Dense(class_n, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)

    return model
