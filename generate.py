import os
import json
import math

import tensorflow as tf
import numpy as np
from PIL import Image

from models import Generator
from utils import set_embedding_chars, concat_imgs, divide_img_dims, save_heatmap

FLAGS = tf.app.flags.FLAGS


def construct_ids(ids):
    ids_x = np.array([], dtype=np.int32)
    ids_y = np.array([], dtype=np.int32)
    ids_alpha = np.array([], dtype=np.float32)
    for id_str in ids:
        if '-' in id_str:
            id_nums = id_str.split('-')
            for i in range(int(id_nums[0]), int(id_nums[1]) + 1):
                ids_x = np.append(ids_x, i)
                ids_y = np.append(ids_y, i)
                ids_alpha = np.append(ids_alpha, 0.)
        elif '*' in id_str:
            id_nums = id_str.split('*')
            for i in range(int(id_nums[1])):
                ids_x = np.append(ids_x, int(id_nums[0]))
                ids_y = np.append(ids_y, int(id_nums[0]))
                ids_alpha = np.append(ids_alpha, 0.)
        elif '..' in id_str and ':' in id_str:
            tmp, step = id_str.split(':')
            id_nums = tmp.split('..')
            for i in range(int(step)):
                ids_x = np.append(ids_x, int(id_nums[0]))
                ids_y = np.append(ids_y, int(id_nums[1]))
                ids_alpha = np.append(ids_alpha, 1. / float(step) * i)
        else:
            ids_x = np.append(ids_x, int(id_str))
            ids_y = np.append(ids_y, int(id_str))
            ids_alpha = np.append(ids_alpha, 0.)
    return ids_x, ids_y, ids_alpha


class GeneratingFontDesignGAN():

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._setup_json()
        self._setup_inputs()
        self._prepare_generating()

    def _setup_dirs(self):
        self.src_log = os.path.join(FLAGS.src_gan, 'log')
        self.dst_generated = os.path.join(FLAGS.src_gan, 'generated')
        if not os.path.exists(self.dst_generated):
            os.mkdir(self.dst_generated)

    def _setup_json(self):
        assert os.path.exists(FLAGS.src_ids), '{} is not found'.format(FLAGS.src_ids)
        with open(FLAGS.src_ids, 'r') as json_file:
            self.json_dict = json.load(json_file)

    def _setup_inputs(self):
        self.font_gen_ids_x, self.font_gen_ids_y, self.font_gen_ids_alpha = construct_ids(self.json_dict['font_ids'])
        self.char_gen_ids_x, self.char_gen_ids_y, self.char_gen_ids_alpha = construct_ids(self.json_dict['char_ids'])
        assert self.font_gen_ids_x.shape[0] == self.char_gen_ids_x.shape[0], \
            'font_ids.shape is not equal char_ids.shape'
        self.batch_size = self.font_gen_ids_x.shape[0]
        self.col_n = self.json_dict['col_n']
        self.row_n = math.ceil(self.batch_size / self.col_n)

    def _prepare_generating(self):
        generator = Generator(img_size=(FLAGS.img_width, FLAGS.img_height),
                              img_dim=FLAGS.img_dim,
                              z_size=FLAGS.z_size,
                              layer_n=4,
                              k_size=3,
                              smallest_hidden_unit_n=64,
                              is_transpose=FLAGS.transpose,
                              is_bn=FLAGS.batchnorm)

        self.font_z_size = int(FLAGS.z_size * FLAGS.font_embedding_rate)
        self.char_z_size = FLAGS.z_size - self.font_z_size
        self.embedding_chars = set_embedding_chars(FLAGS.embedding_chars_type)
        assert self.embedding_chars != [], 'embedding_chars is empty'
        self.char_embedding_n = len(self.embedding_chars)

        font_embedding_np = np.random.uniform(-1, 1, (FLAGS.font_embedding_n, self.font_z_size)).astype(np.float32)
        char_embedding_np = np.random.uniform(-1, 1, (self.char_embedding_n, self.char_z_size)).astype(np.float32)

        with tf.variable_scope('embeddings'):
            font_embedding = tf.Variable(font_embedding_np, name='font_embedding')
            char_embedding = tf.Variable(char_embedding_np, name='char_embedding')
        self.font_ids_x = tf.placeholder(tf.int32, (self.batch_size,), name='font_ids_x')
        self.font_ids_y = tf.placeholder(tf.int32, (self.batch_size,), name='font_ids_y')
        self.font_ids_alpha = tf.placeholder(tf.float32, (self.batch_size,), name='font_ids_alpha')
        self.char_ids_x = tf.placeholder(tf.int32, (self.batch_size,), name='char_ids_x')
        self.char_ids_y = tf.placeholder(tf.int32, (self.batch_size,), name='char_ids_y')
        self.char_ids_alpha = tf.placeholder(tf.float32, (self.batch_size,), name='char_ids_alpha')

        font_z_x = tf.nn.embedding_lookup(font_embedding, self.font_ids_x)
        font_z_y = tf.nn.embedding_lookup(font_embedding, self.font_ids_y)
        font_z = font_z_x * tf.expand_dims(1. - self.font_ids_alpha, 1) + font_z_y * tf.expand_dims(self.font_ids_alpha, 1)
        char_z_x = tf.nn.embedding_lookup(char_embedding, self.char_ids_x)
        char_z_y = tf.nn.embedding_lookup(char_embedding, self.char_ids_y)
        char_z = char_z_x * tf.expand_dims(1. - self.char_ids_alpha, 1) + char_z_y * tf.expand_dims(self.char_ids_alpha, 1)

        z = tf.concat([font_z, char_z], axis=1)

        if FLAGS.intermediate:
            self.generated_imgs, self.intermediate_imgs = generator(z, is_train=False, is_intermediate=True)
        else:
            self.generated_imgs = generator(z, is_train=False)

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)

        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.src_log)
        assert checkpoint, 'cannot get checkpoint: {}'.format(self.src_log)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def generate(self, filename='generated', ext='png'):
        generated_imgs = self.sess.run(self.generated_imgs,
                                       feed_dict={self.font_ids_x: self.font_gen_ids_x,
                                                  self.font_ids_y: self.font_gen_ids_y,
                                                  self.font_ids_alpha: self.font_gen_ids_alpha,
                                                  self.char_ids_x: self.char_gen_ids_x,
                                                  self.char_ids_y: self.char_gen_ids_y,
                                                  self.char_ids_alpha: self.char_gen_ids_alpha})
        concated_img = concat_imgs(generated_imgs, self.row_n, self.col_n)
        concated_img = (concated_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, self.col_n * FLAGS.img_height))
        else:
            concated_img = np.reshape(concated_img, (-1, self.col_n * FLAGS.img_height, FLAGS.img_dim))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(os.path.join(self.dst_generated, '{}.{}'.format(filename, ext)))

    def visualize_intermediate(self, filename='intermediate', ext='png'):
        all_intermediate_imgs = self.sess.run(self.intermediate_imgs,
                                              feed_dict={self.font_ids_x: self.font_gen_ids_x,
                                                         self.font_ids_y: self.font_gen_ids_y,
                                                         self.font_ids_alpha: self.font_gen_ids_alpha,
                                                         self.char_ids_x: self.char_gen_ids_x,
                                                         self.char_ids_y: self.char_gen_ids_y,
                                                         self.char_ids_alpha: self.char_gen_ids_alpha})
        for i, intermediate_imgs in enumerate(all_intermediate_imgs):
            imgs = divide_img_dims(intermediate_imgs)
            # imgs = (imgs - np.mean(imgs)) / np.std(imgs)
            save_heatmap(imgs, 'intermediate layer{}'.format(i), os.path.join(self.dst_generated, '{}_{}.{}'.format(filename, i, ext)))
