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
                    if i == self.layer_n - 1:
                        unit_n = self.img_dim
                    else:
                        unit_n = self.smallest_hidden_unit_n * (2 ** (self.layer_n - i - 2))
                    x_shape = x.get_shape().as_list()
                    new_height = 2 * x_shape[1]
                    new_width = 2 * x_shape[2]
                    x = tf.image.resize_nearest_neighbor(x, (new_height, new_width))
                    x = ops.conv2d(x, unit_n, self.k_size, 1, 'SAME')
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

            unit_n = self.smallest_hidden_unit_n

            for i in range(self.layer_n):
                with tf.variable_scope('layer{}'.format(i + 1)):
                    x = ops.conv2d(x, unit_n, self.k_size, 2, 'SAME')
                    if self.is_bn and i != 0:
                        tf.contrib.layers.batch_norm(x, fused=False, scope='bn')
                    x = ops.lrelu(x)
                    unit_n = self.smallest_hidden_unit_n * (2 ** (i + 1))

            x = tf.reshape(x, (self.batch_size, -1))
            x = ops.linear(x, 1)

            return x


class Classifier(Model):

    def __init__(self, img_size, img_dim, k_size, class_n, smallest_unit_n=64, name='classifier'):
        super(Classifier, self).__init__(name)

        self.img_size = img_size
        self.img_dim = img_dim
        self.k_size = k_size
        self.class_n = class_n
        self.smallest_unit_n = smallest_unit_n
        self.name = name

    def __call__(self, x, is_reuse=False):
        with tf.variable_scope(self.name) as scope:

            if is_reuse:
                scope.reuse_variables()

            unit_n = self.smallest_unit_n
            conv_ns = [2, 2]

            for layer_i, conv_n in enumerate(conv_ns):
                with tf.variable_scope('layer{}'.format(layer_i)):
                    for conv_i in range(conv_n):
                        x = ops.conv2d(x, unit_n, self.k_size, 1, 'SAME', name='conv2d_{}'.format(conv_i))
                        x = tf.nn.relu(x)
                    x = ops.maxpool2d(x, self.k_size, 2, 'SAME')
                unit_n *= 2

            unit_n = 256
            fc_n = 1

            for layer_i in range(len(conv_ns), len(conv_ns) + fc_n):
                with tf.variable_scope('layer{}'.format(layer_i)):
                    x = ops.fc(x, unit_n)
                    x = tf.nn.relu(x)
                    x = tf.contrib.layers.batch_norm(x)
                    x = tf.nn.dropout(x, 0.5)

            with tf.variable_scope('output'.format(layer_i)):
                x = ops.fc(x, self.class_n)

            return x
