import os
import json
import math

import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from dataset import Dataset
from models import GeneratorDCGAN, DiscriminatorDCGAN, GeneratorResNet, DiscriminatorResNet
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
        self._setup_params()
        self._setup_embedding_chars()
        if FLAGS.generate_test or FLAGS.generate_walk:
            # assert FLAGS.char_img_n % FLAGS.batch_size == 0, 'FLAGS.batch_size mod FLAGS.img_n must be 0'
            self.batch_size = FLAGS.batch_size
            while ((FLAGS.char_img_n * self.char_embedding_n) % self.batch_size != 0) or (self.batch_size % self.char_embedding_n != 0):
                self.batch_size -= 1
            print('batch_size: {}'.format(self.batch_size))
            if FLAGS.generate_walk:
                self.walk_step = self.batch_size // self.char_embedding_n
                print('walk_step: {}'.format(self.walk_step))
            self._load_dataset()
        else:
            self._setup_inputs()
        self._prepare_generating()

    def _setup_dirs(self):
        self.src_log = os.path.join(FLAGS.gan_dir, 'log')
        self.dst_generated = os.path.join(FLAGS.gan_dir, 'generated')
        if not os.path.exists(self.dst_generated):
            os.mkdir(self.dst_generated)
        if FLAGS.generate_test:
            self.dst_recognition_test = os.path.join(FLAGS.gan_dir, 'recognition_test')
            if not os.path.exists(self.dst_recognition_test):
                os.makedirs(self.dst_recognition_test)
        if FLAGS.generate_walk:
            self.dst_walk = os.path.join(FLAGS.gan_dir, 'random_walking')
            if not os.path.exists(self.dst_walk):
                os.makedirs(self.dst_walk)
        if FLAGS.intermediate:
            self.dst_intermediate = os.path.join(FLAGS.gan_dir, 'intermediate')
            if not os.path.exists(self.dst_intermediate):
                os.makedirs(self.dst_intermediate)

    def _setup_params(self):
        with open(os.path.join(self.src_log, 'flags.json'), 'r') as json_file:
            json_dict = json.load(json_file)
        keys = ['embedding_chars_type', 'img_width', 'img_height', 'img_dim', 'font_z_size', 'font_h5',
                'font_embedding_n', 'batchnorm', 'transpose']
        for key in keys:
            setattr(self, key, json_dict[key])

    def _setup_embedding_chars(self):
        self.embedding_chars = set_embedding_chars(self.embedding_chars_type)
        assert self.embedding_chars != [], 'embedding_chars is empty'
        self.char_embedding_n = len(self.embedding_chars)

    def _setup_inputs(self):
        assert os.path.exists(FLAGS.src_ids), '{} is not found'.format(FLAGS.src_ids)
        with open(FLAGS.src_ids, 'r') as json_file:
            json_dict = json.load(json_file)
        self.font_gen_ids_x, self.font_gen_ids_y, self.font_gen_ids_alpha = construct_ids(json_dict['font_ids'])
        self.char_gen_ids_x, self.char_gen_ids_y, self.char_gen_ids_alpha = construct_ids(json_dict['char_ids'])
        assert self.font_gen_ids_x.shape[0] == self.char_gen_ids_x.shape[0], \
            'font_ids.shape is not equal char_ids.shape'
        self.batch_size = self.font_gen_ids_x.shape[0]
        self.col_n = json_dict['col_n']
        self.row_n = math.ceil(self.batch_size / self.col_n)

    def _load_dataset(self):
        self.real_dataset = Dataset(self.font_h5, 'r', self.img_width, self.img_height, self.img_dim)
        self.real_dataset.set_load_data()

    def _prepare_generating(self):
        self.z_size = self.font_z_size + self.char_embedding_n

        if FLAGS.arch == 'DCGAN':
            generator = GeneratorDCGAN(img_size=(FLAGS.img_width, FLAGS.img_height),
                                       img_dim=FLAGS.img_dim,
                                       z_size=self.z_size,
                                       layer_n=4,
                                       k_size=3,
                                       smallest_hidden_unit_n=64,
                                       is_transpose=FLAGS.transpose,
                                       is_bn=FLAGS.batchnorm)
            discriminator = DiscriminatorDCGAN(img_size=(FLAGS.img_width, FLAGS.img_height),
                                               img_dim=FLAGS.img_dim,
                                               layer_n=4,
                                               k_size=3,
                                               smallest_hidden_unit_n=64,
                                               is_bn=FLAGS.batchnorm)
        elif FLAGS.arch == 'ResNet':
            generator = GeneratorResNet(k_size=3, smallest_unit_n=64)
            discriminator = DiscriminatorResNet(k_size=3, smallest_unit_n=64)

        if FLAGS.generate_test:
            font_embedding_np = np.random.uniform(-1, 1, (FLAGS.char_img_n, self.font_z_size)).astype(np.float32)
        elif FLAGS.generate_walk:
            font_embedding_np = np.random.uniform(-1, 1, (FLAGS.char_img_n // self.walk_step, self.font_z_size)).astype(np.float32)
        else:
            font_embedding_np = np.random.uniform(-1, 1, (self.font_embedding_n, self.font_z_size)).astype(np.float32)

        with tf.variable_scope('embeddings'):
            font_embedding = tf.Variable(font_embedding_np, name='font_embedding')
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
        char_z_x = tf.one_hot(self.char_ids_x, self.char_embedding_n)
        char_z_y = tf.one_hot(self.char_ids_y, self.char_embedding_n)
        char_z = char_z_x * tf.expand_dims(1. - self.char_ids_alpha, 1) \
            + char_z_y * tf.expand_dims(self.char_ids_alpha, 1)

        z = tf.concat([font_z, char_z], axis=1)

        if FLAGS.intermediate:
            self.generated_imgs, gen_intermediates = generator(z, is_train=False, is_intermediate=True)
            _, disc_intermediates = discriminator(self.generated_imgs, is_train=False, is_intermediate=True)
            self.intermediates = list()
            self.intermediate_names = list()

            self.intermediates.append(z)
            self.intermediate_names.append('z')
            for i, intermediate in enumerate(gen_intermediates):
                self.intermediates.append(intermediate)
                self.intermediate_names.append('gen{}'.format(i))
            reshaped_generated_imgs = tf.reshape(self.generated_imgs, [self.batch_size, -1])
            self.intermediates.append(reshaped_generated_imgs)
            self.intermediate_names.append('pic')
            for i, intermediate in enumerate(disc_intermediates):
                self.intermediates.append(intermediate)
                self.intermediate_names.append('disc{}'.format(i))
        else:
            self.generated_imgs = generator(z, is_train=False)

        if FLAGS.intermediate or FLAGS.gpu_ids == "":
            sess_config = tf.ConfigProto(
                device_count={"GPU": 0},
                log_device_placement=True
            )
        else:
            sess_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
            )
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        if FLAGS.generate_test or FLAGS.generate_walk:
            var_list = [var for var in tf.global_variables() if 'embedding' not in var.name]
        else:
            var_list = [var for var in tf.global_variables()]
        pretrained_saver = tf.train.Saver(var_list=var_list)
        checkpoint = tf.train.get_checkpoint_state(self.src_log)
        assert checkpoint, 'cannot get checkpoint: {}'.format(self.src_log)
        pretrained_saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def _concat_and_save_imgs(self, src_imgs, dst_path):
        concated_img = concat_imgs(src_imgs, self.row_n, self.col_n)
        concated_img = (concated_img + 1.) * 127.5
        if self.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, self.col_n * self.img_height))
        else:
            concated_img = np.reshape(concated_img, (-1, self.col_n * self.img_height, self.img_dim))
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
        for c in self.embedding_chars:
            dst_dir = os.path.join(self.dst_recognition_test, c)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
        batch_n = (self.char_embedding_n * FLAGS.char_img_n) // self.batch_size
        c_ids = self.real_dataset.get_ids_from_labels(self.embedding_chars)
        for batch_i in tqdm(range(batch_n)):
            batch_font_n = self.batch_size // self.char_embedding_n
            font_id_start = batch_i * batch_font_n
            font_id_end = (batch_i + 1) * batch_font_n
            generated_imgs = self.sess.run(self.generated_imgs,
                                           feed_dict={self.font_ids_x: np.repeat(np.arange(font_id_start, font_id_end), self.char_embedding_n),
                                                      self.font_ids_y: np.repeat(np.arange(font_id_start, font_id_end), self.char_embedding_n),
                                                      self.font_ids_alpha: np.zeros(self.batch_size),
                                                      self.char_ids_x: np.tile(c_ids, self.batch_size // self.char_embedding_n),
                                                      self.char_ids_y: np.tile(c_ids, self.batch_size // self.char_embedding_n),
                                                      self.char_ids_alpha: np.zeros(self.batch_size)})
            for img_i in range(generated_imgs.shape[0]):
                img = generated_imgs[img_i]
                img = (img + 1.) * 127.5
                pil_img = Image.fromarray(np.uint8(img))
                pil_img.save(os.path.join(self.dst_recognition_test,
                             str(self.embedding_chars[img_i % self.char_embedding_n]),
                             '{:05d}.png'.format(font_id_start + img_i // self.char_embedding_n)))

    def generate_random_walking(self):
        for c in self.embedding_chars:
            dst_dir = os.path.join(self.dst_walk, c)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
        batch_n = (self.char_embedding_n * FLAGS.char_img_n) // self.batch_size
        c_ids = self.real_dataset.get_ids_from_labels(self.embedding_chars)
        for batch_i in tqdm(range(batch_n)):
            font_id_start = batch_i
            if batch_i == batch_n - 1:
                font_id_end = 0
            else:
                font_id_end = batch_i + 1
            generated_imgs = self.sess.run(self.generated_imgs,
                                           feed_dict={self.font_ids_x: np.ones(self.batch_size) * font_id_start,
                                                      self.font_ids_y: np.ones(self.batch_size) * font_id_end,
                                                      self.font_ids_alpha: np.repeat(np.linspace(0., 1., num=self.walk_step, endpoint=False), self.char_embedding_n),
                                                      self.char_ids_x: np.tile(c_ids, self.batch_size // self.char_embedding_n),
                                                      self.char_ids_y: np.tile(c_ids, self.batch_size // self.char_embedding_n),
                                                      self.char_ids_alpha: np.zeros(self.batch_size)})
            for img_i in range(generated_imgs.shape[0]):
                img = generated_imgs[img_i]
                img = (img + 1.) * 127.5
                pil_img = Image.fromarray(np.uint8(img))
                pil_img.save(os.path.join(self.dst_walk,
                             str(self.embedding_chars[img_i % self.char_embedding_n]),
                             '{:05d}.png'.format((batch_i * self.batch_size + img_i) // self.char_embedding_n)))

    def visualize_intermediate(self, filename='intermediate', is_tensorboard=True, is_plot=True):
        rets = \
            self.sess.run([self.generated_imgs] + [self.intermediates[i] for i in range(len(self.intermediates))],
                          feed_dict={self.font_ids_x: self.font_gen_ids_x,
                                     self.font_ids_y: self.font_gen_ids_y,
                                     self.font_ids_alpha: self.font_gen_ids_alpha,
                                     self.char_ids_x: self.char_gen_ids_x,
                                     self.char_ids_y: self.char_gen_ids_y,
                                     self.char_ids_alpha: self.char_gen_ids_alpha})
        dst_path = os.path.join(self.dst_intermediate, '{}.png'.format(filename))
        for ret in rets:
            print(ret.shape)
        self._concat_and_save_imgs(rets[0], dst_path)
        if is_plot:
            self._plot_tsne(rets[1:], filename)

    def _plot_tsne(self, intermediates, filename):
        font_n = self.batch_size // 26
        char_labels = [chr(i + 65) for i in np.tile(np.arange(0, 26), font_n).tolist()]
        style_labels = np.repeat(np.arange(0, font_n), 26)
        for intermediate_i, (intermediate, intermediate_name) in enumerate(zip(intermediates, self.intermediate_names)):
            if FLAGS.plot_method == 'MDS':
                reduced = MDS(n_components=2, verbose=3, n_jobs=8).fit_transform(intermediate)
                method_name = 'MDS'
            elif FLAGS.plot_method == 'PCA':
                reduced = PCA(n_components=2).fit_transform(intermediate)
                method_name = 'PCA'
            else:
                reduced = TSNE(n_components=2, verbose=3, perplexity=FLAGS.tsne_p, n_jobs=8,
                               n_iter=5000).fit_transform(intermediate)
                method_name = 'TSNE({})'.format(FLAGS.tsne_p)
            plt.figure(figsize=(16, 9))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=["w" for i in char_labels])
            cmap = plt.get_cmap('jet')
            splitted = list()
            for i in range(reduced.shape[0]):
                splitted.append(cmap(style_labels[i] / font_n))
                plt.text(reduced[i][0], reduced[i][1], char_labels[i],
                         fontdict={'size': 10, 'color': splitted[i]})
                splitted_cmap = ListedColormap(splitted)
            sm = plt.cm.ScalarMappable(cmap=splitted_cmap, norm=plt.Normalize(vmin=-0.5, vmax=font_n - 0.5))
            sm._A = []
            cbar = plt.colorbar(sm, ticks=[i for i in range(font_n + 1)])
            cbar.ax.invert_yaxis()
            plt.savefig(os.path.join(self.dst_intermediate,
                                     '{}_{}_{}_{}.png'.format(filename, method_name, intermediate_i, intermediate_name)))
            plt.close()
