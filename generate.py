import numpy as np
import os
from PIL import Image
import tensorflow as tf

import models
from utils import concat_imgs

FLAGS = tf.app.flags.FLAGS


class GeneratingFontDesignGAN():

    def __init__(self):
        global FLAGS

    def setup(self):
        self._make_dirs()
        self._build_model()
        self._prepare_generating()

    def _make_dirs(self):
        os.mkdir(FLAGS.dst_root)

    def _build_model(self):
        self.generator = models.Generator(img_size=(FLAGS.img_width, FLAGS.img_height),
                                          img_dim=FLAGS.img_dim,
                                          z_size=FLAGS.z_size,
                                          layer_n=FLAGS.g_layer_n,
                                          k_size=FLAGS.g_k_size,
                                          smallest_hidden_unit_n=FLAGS.g_smallest_hidden_unit_n)

    def _prepare_generating(self):
        self.font_z_size = int(FLAGS.z_size * FLAGS.font_embedding_rate)
        self.char_z_size = FLAGS.z_size - self.font_z_size

        font_embedding_np = np.random.uniform(-1, 1, (FLAGS.font_embedding_n, self.font_z_size)).astype(np.float32)
        char_embedding_np = np.random.uniform(-1, 1, (FLAGS.char_embedding_n, self.char_z_size)).astype(np.float32)

        with tf.variable_scope('embeddings'):
            font_embedding = tf.Variable(font_embedding_np, name='font_embedding')
            char_embedding = tf.Variable(char_embedding_np, name='char_embedding')
        self.font_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='font_ids')
        self.char_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='char_ids')

        font_z = tf.nn.embedding_lookup(font_embedding, self.font_ids)
        char_z = tf.nn.embedding_lookup(char_embedding, self.char_ids)

        z = tf.concat([font_z, char_z], axis=1)

        self.generated_imgs = self.generator(z)

        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, FLAGS.src_ckpt)

    def generate(self, filename='generated.png'):
        font_ids = np.random.randint(0, FLAGS.font_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        char_ids = np.random.randint(0, FLAGS.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        generated_imgs = self.sess.run(self.generated_imgs,
                                       feed_dict={self.font_ids: font_ids,
                                                  self.char_ids: char_ids})
        concated_img = concat_imgs(generated_imgs, 16, 16)
        concated_img = (concated_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, 16 * FLAGS.img_height))
        else:
            concated_img = np.reshape(concated_img, (-1, 16 * FLAGS.img_height, FLAGS.img_dim))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(os.path.join(FLAGS.dst_root, filename))
