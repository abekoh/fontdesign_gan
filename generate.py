import numpy as np
import os
from PIL import Image
import tensorflow as tf
import json
import math

import models
from utils import concat_imgs

FLAGS = tf.app.flags.FLAGS


class GeneratingFontDesignGAN():

    def __init__(self, json_path):
        global FLAGS
        if json_path is not None:
            with open(json_path, 'r') as json_file:
                self.json_dict = json.load(json_file)

    def setup(self):
        self._make_dirs()
        self._build_model()
        self._set_inputs()
        self._prepare_generating()

    def _make_dirs(self):
        if not os.path.exists(FLAGS.dst_gen_root):
            os.mkdir(FLAGS.dst_gen_root)

    def _build_model(self):
        self.generator = models.Generator(img_size=(FLAGS.img_width, FLAGS.img_height),
                                          img_dim=FLAGS.img_dim,
                                          z_size=FLAGS.z_size,
                                          layer_n=FLAGS.g_layer_n,
                                          k_size=FLAGS.g_k_size,
                                          smallest_hidden_unit_n=FLAGS.g_smallest_hidden_unit_n)

    def _set_inputs(self):
        self.font_gen_ids = self._construct_ids('font_ids')
        self.char_gen_ids = self._construct_ids('char_ids')
        assert self.font_gen_ids.shape[0] == self.char_gen_ids.shape[0], \
            'font_ids.shape is not equal char_ids.shape'
        self.batch_size = self.font_gen_ids.shape[0]
        self.col_n = self.json_dict['col_n']
        self.row_n = math.ceil(self.batch_size / self.col_n)

    def _construct_ids(self, label):
        ids = np.array([], dtype=np.int32)
        for id_str in self.json_dict[label]:
            if '-' in id_str:
                id_nums = id_str.split('-')
                for i in range(int(id_nums[0]), int(id_nums[1]) + 1):
                    ids = np.append(ids, i)
            elif '*' in id_str:
                id_nums = id_str.split('*')
                for i in range(int(id_nums[1])):
                    ids = np.append(ids, int(id_nums[0]))
            else:
                ids.append(int(id_str))
        return np.array(ids, dtype=np.int32)

    def _prepare_generating(self):
        self.font_z_size = int(FLAGS.z_size * FLAGS.font_embedding_rate)
        self.char_z_size = FLAGS.z_size - self.font_z_size

        font_embedding_np = np.random.uniform(-1, 1, (FLAGS.font_embedding_n, self.font_z_size)).astype(np.float32)
        char_embedding_np = np.random.uniform(-1, 1, (FLAGS.char_embedding_n, self.char_z_size)).astype(np.float32)

        with tf.variable_scope('embeddings'):
            font_embedding = tf.Variable(font_embedding_np, name='font_embedding')
            char_embedding = tf.Variable(char_embedding_np, name='char_embedding')
        self.font_ids = tf.placeholder(tf.int32, (self.batch_size,), name='font_ids')
        self.char_ids = tf.placeholder(tf.int32, (self.batch_size,), name='char_ids')

        font_z = tf.nn.embedding_lookup(font_embedding, self.font_ids)
        char_z = tf.nn.embedding_lookup(char_embedding, self.char_ids)

        z = tf.concat([font_z, char_z], axis=1)

        self.generated_imgs = self.generator(z)

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, FLAGS.src_trained_ckpt)

    def generate(self, filename='generated.png'):
        generated_imgs = self.sess.run(self.generated_imgs,
                                       feed_dict={self.font_ids: self.font_gen_ids,
                                                  self.char_ids: self.char_gen_ids})
        concated_img = concat_imgs(generated_imgs, self.row_n, self.col_n)
        concated_img = (concated_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, self.col_n * FLAGS.img_height))
        else:
            concated_img = np.reshape(concated_img, (-1, self.col_n * FLAGS.img_height, FLAGS.img_dim))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(os.path.join(FLAGS.dst_gen_root, filename))
