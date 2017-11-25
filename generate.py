import os
import json
import math

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import Dataset
from models import Generator, Discriminator
from utils import set_embedding_chars, concat_imgs

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
        if FLAGS.recogtest:
            assert FLAGS.char_img_n % FLAGS.batch_size == 0, 'FLAGS.batch_size mod FLAGS.img_n must be 0'
            self.batch_size = FLAGS.batch_size
            self._load_dataset()
        else:
            self._setup_json()
            self._setup_inputs()
        self._prepare_generating()

    def _setup_dirs(self):
        self.src_log = os.path.join(FLAGS.src_gan, 'log')
        self.dst_generated = os.path.join(FLAGS.src_gan, 'generated')
        if not os.path.exists(self.dst_generated):
            os.mkdir(self.dst_generated)
        if FLAGS.recogtest:
            self.dst_recognition_test = os.path.join(FLAGS.src_gan, 'recognition_test')
            if not os.path.exists(self.dst_recognition_test):
                os.makedirs(self.dst_recognition_test)
        if FLAGS.mode == 'intermediate':
            self.dst_intermediate = os.path.join(FLAGS.src_gan, 'intermediate')
            if not os.path.exists(self.dst_intermediate):
                os.makedirs(self.dst_intermediate)

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

    def _load_dataset(self):
        self.real_dataset = Dataset(FLAGS.font_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.real_dataset.set_load_data()

    def _prepare_generating(self):
        generator = Generator(img_size=(FLAGS.img_width, FLAGS.img_height),
                              img_dim=FLAGS.img_dim,
                              z_size=FLAGS.z_size,
                              layer_n=4,
                              k_size=3,
                              smallest_hidden_unit_n=64,
                              is_transpose=FLAGS.transpose,
                              is_bn=FLAGS.batchnorm)

        discriminator = Discriminator(img_size=(FLAGS.img_width, FLAGS.img_height),
                                      img_dim=FLAGS.img_dim,
                                      layer_n=4,
                                      k_size=3,
                                      smallest_hidden_unit_n=64,
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

        # If sum of (font/char)_ids is less than -1, z is generated from uniform distribution
        font_z_x = tf.cond(tf.less(tf.reduce_sum(self.font_ids_x), 0),
                           lambda: tf.random_uniform((self.batch_size, self.font_z_size), -1, 1),
                           lambda: tf.nn.embedding_lookup(font_embedding, self.font_ids_x))
        font_z_y = tf.cond(tf.less(tf.reduce_sum(self.font_ids_y), 0),
                           lambda: tf.random_uniform((self.batch_size, self.font_z_size), -1, 1),
                           lambda: tf.nn.embedding_lookup(font_embedding, self.font_ids_y))
        font_z = font_z_x * tf.expand_dims(1. - self.font_ids_alpha, 1) \
            + font_z_y * tf.expand_dims(self.font_ids_alpha, 1)
        char_z_x = tf.cond(tf.less(tf.reduce_sum(self.char_ids_x), 0),
                           lambda: tf.random_uniform((self.batch_size, self.char_z_size), -1, 1),
                           lambda: tf.nn.embedding_lookup(char_embedding, self.char_ids_x))
        char_z_y = tf.cond(tf.less(tf.reduce_sum(self.char_ids_y), 0),
                           lambda: tf.random_uniform((self.batch_size, self.char_z_size), -1, 1),
                           lambda: tf.nn.embedding_lookup(char_embedding, self.char_ids_y))
        char_z = char_z_x * tf.expand_dims(1. - self.char_ids_alpha, 1) \
            + char_z_y * tf.expand_dims(self.char_ids_alpha, 1)

        z = tf.concat([font_z, char_z], axis=1)

        if FLAGS.mode == 'intermediate':
            self.generated_imgs, self.gen_intermediates = generator(z, is_train=False, is_intermediate=True)
            _, self.disc_intermediates = discriminator(self.generated_imgs, is_train=False, is_intermediate=True)
            self.intermediates = list()
            with tf.variable_scope('intermediate/gen'):
                for i, intermediate in enumerate(self.gen_intermediates):
                    self.intermediates.append(tf.Variable(intermediate, name='{}'.format(i)))
            with tf.variable_scope('intermediate/disc'):
                for i, intermediate in enumerate(self.disc_intermediates):
                    self.intermediates.append(tf.Variable(intermediate, name='{}'.format(i)))
        else:
            self.generated_imgs = generator(z, is_train=False)

        if FLAGS.gpu_ids == "":
            sess_config = tf.ConfigProto(
                device_count={"GPU": 0},
                log_device_placement=True
            )
        else:
            sess_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
            )
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer(),
                      feed_dict={self.font_ids_x: self.font_gen_ids_x,
                                 self.font_ids_y: self.font_gen_ids_y,
                                 self.font_ids_alpha: self.font_gen_ids_alpha,
                                 self.char_ids_x: self.char_gen_ids_x,
                                 self.char_ids_y: self.char_gen_ids_y,
                                 self.char_ids_alpha: self.char_gen_ids_alpha})

        var_list = [var for var in tf.global_variables() if 'intermediate' not in var.name]
        pretrained_saver = tf.train.Saver(var_list=var_list)
        checkpoint = tf.train.get_checkpoint_state(self.src_log)
        assert checkpoint, 'cannot get checkpoint: {}'.format(self.src_log)
        pretrained_saver.restore(self.sess, checkpoint.model_checkpoint_path)

        if FLAGS.mode == 'intermediate':
            intermediate_vars = [var for var in tf.global_variables() if 'intermediate' in var.name]
            self.saver = tf.train.Saver(var_list=intermediate_vars)
            self.writer = tf.summary.FileWriter(self.dst_intermediate)

    def _concat_and_save_imgs(self, src_imgs, dst_path):
        concated_img = concat_imgs(src_imgs, self.row_n, self.col_n)
        concated_img = (concated_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, self.col_n * FLAGS.img_height))
        else:
            concated_img = np.reshape(concated_img, (-1, self.col_n * FLAGS.img_height, FLAGS.img_dim))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(dst_path)

    def generate(self, filename='generated'):
        generated_imgs = self.sess.run(self.generated_imgs,
                                       feed_dict={self.font_ids_x: self.font_gen_ids_x,
                                                  self.font_ids_y: self.font_gen_ids_y,
                                                  self.font_ids_alpha: self.font_gen_ids_alpha,
                                                  self.char_ids_x: self.char_gen_ids_x,
                                                  self.char_ids_y: self.char_gen_ids_y,
                                                  self.char_ids_alpha: self.char_gen_ids_alpha})
        self._concat_and_save_imgs(generated_imgs, os.path.join(self.dst_generated, '{}.png'.format(filename)))

    def generate_for_recognition_test(self):
        for c in tqdm(self.embedding_chars):
            dst_dir = os.path.join(self.dst_recognition_test, c)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            c_id = self.real_dataset.get_ids_from_labels([c])[0]
            for batch_i in range(FLAGS.char_img_n // self.batch_size):
                generated_imgs = self.sess.run(self.generated_imgs,
                                               feed_dict={self.font_ids_x: np.ones(self.batch_size) * -1,
                                                          self.font_ids_y: np.ones(self.batch_size) * -1,
                                                          self.font_ids_alpha: np.zeros(self.batch_size),
                                                          self.char_ids_x: np.repeat(c_id, self.batch_size),
                                                          self.char_ids_y: np.repeat(c_id, self.batch_size),
                                                          self.char_ids_alpha: np.zeros(self.batch_size)})
                for img_i in range(generated_imgs.shape[0]):
                    img = generated_imgs[img_i]
                    img = (img + 1.) * 127.5
                    pil_img = Image.fromarray(np.uint8(img))
                    pil_img.save(os.path.join(dst_dir, '{}.png'.format(batch_i * self.batch_size + img_i)))

    def _extract_intermediate(self, filename='intermediate'):
        generated_imgs, gen_intermediates, disc_intermediates = \
            self.sess.run([self.generated_imgs, self.gen_intermediates, self.disc_intermediates],
                          feed_dict={self.font_ids_x: self.font_gen_ids_x,
                                     self.font_ids_y: self.font_gen_ids_y,
                                     self.font_ids_alpha: self.font_gen_ids_alpha,
                                     self.char_ids_x: self.char_gen_ids_x,
                                     self.char_ids_y: self.char_gen_ids_y,
                                     self.char_ids_alpha: self.char_gen_ids_alpha})
        dst_path = os.path.join(self.dst_intermediate, '{}.png'.format(filename))
        self._concat_and_save_imgs(generated_imgs, dst_path)
        return gen_intermediates, disc_intermediates, os.path.realpath(dst_path)

    def project_tensorboard(self, filename='intermediate'):
        _, _, img_path = self._extract_intermediate(filename)
        ckpt_path = os.path.join(self.dst_intermediate, 'result.ckpt')
        self.saver.save(self.sess, ckpt_path)
        summary_writer = tf.summary.FileWriter(self.dst_intermediate)
        config = projector.ProjectorConfig()
        config.model_checkpoint_path = ckpt_path
        for intermediate in self.intermediates:
            embedding_config = config.embeddings.add()
            embedding_config.tensor_name = intermediate.name
            embedding_config.sprite.image_path = img_path
            embedding_config.sprite.single_image_dim.extend([FLAGS.img_width, FLAGS.img_height])
        projector.visualize_embeddings(summary_writer, config)
