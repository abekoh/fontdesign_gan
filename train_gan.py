import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from subprocess import Popen, PIPE

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from keras.utils import to_categorical

import models
from dataset import Dataset
from utils import concat_imgs

FLAGS = tf.app.flags.FLAGS


class TrainingFontDesignGAN():

    def __init__(self):
        global FLAGS

    def setup(self):
        self._make_dirs()
        self._build_models()
        self._load_dataset()
        self._prepare_training()

    def __del__(self):
        Popen(['killall', 'tensorbaord'], stdout=PIPE)

    def reset(self):
        tf.reset_default_graph()

    def _make_dirs(self):
        os.mkdir(FLAGS.dst_train_root)
        os.mkdir(FLAGS.dst_train_log)
        os.mkdir(FLAGS.dst_train_samples)

    def _build_models(self):
        self.generator = models.Generator(img_size=(FLAGS.img_width, FLAGS.img_height),
                                          img_dim=FLAGS.img_dim,
                                          z_size=FLAGS.z_size,
                                          layer_n=FLAGS.g_layer_n,
                                          k_size=FLAGS.g_k_size,
                                          smallest_hidden_unit_n=FLAGS.g_smallest_hidden_unit_n)
        self.discriminator = models.Discriminator(img_size=(FLAGS.img_width, FLAGS.img_height),
                                                  img_dim=FLAGS.img_dim,
                                                  layer_n=FLAGS.d_layer_n,
                                                  k_size=FLAGS.d_k_size,
                                                  smallest_hidden_unit_n=FLAGS.d_smallest_hidden_unit_n)
        if FLAGS.c_penalty > 0.:
            self.classifier = models.Classifier(img_size=(FLAGS.img_width, FLAGS.img_height),
                                                img_dim=FLAGS.img_dim,
                                                k_size=FLAGS.c_k_size,
                                                class_n=26,
                                                smallest_unit_n=FLAGS.c_smallest_unit_n)

    def _load_dataset(self, is_shuffle=True):
        self.real_dataset = Dataset(FLAGS.src_real_h5, 'r', img_size=(FLAGS.img_width, FLAGS.img_height), img_dim=FLAGS.img_dim)
        self.real_dataset.set_load_data()
        if is_shuffle:
            self.real_dataset.shuffle()
        self.real_data_n = self.real_dataset.get_img_len()

    def _prepare_training(self):
        self.font_z_size = int(FLAGS.z_size * FLAGS.font_embedding_rate)
        self.char_z_size = FLAGS.z_size - self.font_z_size

        font_embedding_np = np.random.uniform(-1, 1, (FLAGS.font_embedding_n, self.font_z_size)).astype(np.float32)
        char_embedding_np = np.random.uniform(-1, 1, (FLAGS.char_embedding_n, self.char_z_size)).astype(np.float32)

        with tf.variable_scope('embeddings'):
            self.font_embedding = tf.Variable(font_embedding_np, name='font_embedding')
            self.char_embedding = tf.Variable(char_embedding_np, name='char_embedding')

        self.font_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='font_ids')
        self.char_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='char_ids')
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        font_z = tf.cond(tf.less(tf.reduce_sum(self.font_ids), 0),
                         lambda: tf.random_uniform((FLAGS.batch_size, self.font_z_size), -1, 1),
                         lambda: tf.nn.embedding_lookup(self.font_embedding, self.font_ids))
        char_z = tf.cond(tf.less(tf.reduce_sum(self.char_ids), 0),
                         lambda: tf.random_uniform((FLAGS.batch_size, self.char_z_size), -1, 1),
                         lambda: tf.nn.embedding_lookup(self.char_embedding, self.char_ids))
        z = tf.concat([font_z, char_z], axis=1)

        self.real_imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='real_imgs')
        self.fake_imgs = self.generator(z, is_train=self.is_train)

        self.d_real = self.discriminator(self.real_imgs, is_train=self.is_train)
        self.d_fake = self.discriminator(self.fake_imgs, is_reuse=True, is_train=self.is_train)

        self.d_loss = - (tf.reduce_mean(self.d_real) - tf.reduce_mean(self.d_fake))
        self.g_loss = - tf.reduce_mean(self.d_fake)

        epsilon = tf.random_uniform((FLAGS.batch_size, 1, 1, 1), minval=0., maxval=1.)
        interp = self.real_imgs + epsilon * (self.fake_imgs - self.real_imgs)
        d_interp = self.discriminator(interp, is_reuse=True, is_train=self.is_train)
        grads = tf.gradients(d_interp, [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[-1]))
        self.grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += 10 * self.grad_penalty

        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('g_loss', self.g_loss)

        d_vars = self.discriminator.get_trainable_variables()
        g_vars = self.generator.get_trainable_variables()

        self.d_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_vars)
        self.g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_vars)

        if FLAGS.c_penalty > 0.:
            self.labels = tf.placeholder(tf.float32, (None, FLAGS.char_embedding_n))
            self.c_fake = FLAGS.c_penalty * self.classifier(self.fake_imgs, is_train=self.is_train)
            self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.c_fake))
            tf.summary.scalar('c_loss', self.c_loss)
            self.c_opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.c_lr).minimize(self.c_loss, var_list=g_vars)
            correct_pred = tf.equal(tf.argmax(self.c_fake, 1), tf.argmax(self.labels, 1))
            self.c_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('c_acc', self.c_acc)
            c_vars = [var for var in tf.global_variables() if 'classifier' in var.name]

        self.summary = tf.summary.merge_all()

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        self.saver_pretrained = tf.train.Saver(var_list=c_vars)
        self.saver_pretrained.restore(self.sess, FLAGS.src_classifier_ckpt)

        self.saver = tf.train.Saver()

        self.writer = tf.summary.FileWriter(FLAGS.dst_train_log)

    def _get_ids(self, is_embedding_font_ids, is_embedding_char_ids):
        if is_embedding_font_ids:
            font_ids = np.random.randint(0, FLAGS.font_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        else:
            font_ids = np.ones(FLAGS.batch_size) * -1
        if is_embedding_char_ids:
            char_ids = np.random.randint(0, FLAGS.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        else:
            char_ids = np.ones(FLAGS.batch_size) * -1
        return font_ids, char_ids

    def train(self):

        if FLAGS.is_run_tensorboard:
            self._run_tensorboard()

        batch_n = self.real_data_n // FLAGS.batch_size

        for epoch_i in tqdm(range(FLAGS.epoch_n)):

            for batch_i in tqdm(range(batch_n)):

                count_i = epoch_i * batch_n + batch_i

                for i in range(FLAGS.critic_n):

                    real_imgs, _ = self.real_dataset.get_random(FLAGS.batch_size)
                    font_ids, char_ids = self._get_ids(False, False)

                    self.sess.run(self.d_opt, feed_dict={self.font_ids: font_ids,
                                                         self.char_ids: char_ids,
                                                         self.real_imgs: real_imgs,
                                                         self.is_train: True})

                font_ids, char_ids = self._get_ids(False, False)

                self.sess.run(self.g_opt, feed_dict={self.font_ids: font_ids,
                                                     self.char_ids: char_ids,
                                                     self.is_train: True})

                if FLAGS.c_penalty > 0.:
                    font_ids, char_ids = self._get_ids(False, True)
                    batched_labels = to_categorical(char_ids, FLAGS.char_embedding_n)
                    self.sess.run(self.c_opt, feed_dict={self.font_ids: font_ids,
                                                         self.char_ids: char_ids,
                                                         self.labels: batched_labels,
                                                         self.is_train: True})

                summary = self.sess.run(self.summary,
                                        feed_dict={self.font_ids: font_ids,
                                                   self.char_ids: char_ids,
                                                   self.labels: batched_labels,
                                                   self.real_imgs: real_imgs,
                                                   self.is_train: True})

                self.writer.add_summary(summary, count_i)

                # save images
                if (batch_i + 1) % FLAGS.save_imgs_interval == 0:
                    self.save_temp_imgs(os.path.join(FLAGS.dst_train_samples, '{}_{}.png'.format(epoch_i + 1, batch_i + 1)))

            self.saver.save(self.sess, os.path.join(FLAGS.dst_train_log, 'result_{}.ckpt'.format(epoch_i)))
            self._visualize_embedding(epoch_i)

    def _run_tensorboard(self):
        Popen(['tensorboard', '--logdir', '{}'.format(os.path.realpath(FLAGS.dst_train_log))], stdout=PIPE)

    def _generate_img(self, font_ids, char_ids, row_n, col_n):
        batched_generated_imgs = self.sess.run(self.fake_imgs, feed_dict={self.font_ids: font_ids,
                                                                          self.char_ids: char_ids,
                                                                          self.is_train: False})
        concated_img = concat_imgs(batched_generated_imgs, row_n, col_n)
        concated_img = (concated_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, col_n * FLAGS.img_height))
        else:
            concated_img = np.reshape(concated_img, (-1, col_n * FLAGS.img_height, FLAGS.img_dim))
        return concated_img

    def _init_temp_imgs_inputs(self):
        self.temp_font_ids = np.random.randint(0, FLAGS.font_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        self.temp_char_ids = np.random.randint(0, FLAGS.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)

    def save_temp_imgs(self, filepath):
        if not hasattr(self, 'temp_font_ids'):
            self._init_temp_imgs_inputs()
        concated_img = self._generate_img(self.temp_font_ids, self.temp_char_ids,
                                          FLAGS.save_imgs_col_n, FLAGS.save_imgs_col_n)
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(filepath)

    def _init_visualize_imgs_inputs(self):
        self.vis_font_ids = np.arange(0, FLAGS.font_embedding_n, dtype=np.int32)
        self.vis_char_ids = np.repeat(np.array([0], dtype=np.int32), FLAGS.font_embedding_n)

    def _visualize_embedding(self, epoch_i):
        if not hasattr(self, 'vis_font_ids'):
            self._init_visualize_imgs_inputs()
        vis_img_path = os.path.realpath(os.path.join(FLAGS.dst_train_log, 'vis_{}.png'.format(epoch_i)))

        vis_img = self._generate_img(self.vis_font_ids, self.vis_char_ids,
                                     FLAGS.save_imgs_col_n, FLAGS.save_imgs_col_n)
        vis_img = Image.fromarray(np.uint8(vis_img))
        vis_img.save(vis_img_path)

        summary_writer = tf.summary.FileWriter(FLAGS.dst_train_log)
        config = projector.ProjectorConfig()
        font_embedding = config.embeddings.add()
        font_embedding.tensor_name = 'embeddings/font_embedding'
        font_embedding.sprite.image_path = vis_img_path
        font_embedding.sprite.single_image_dim.extend([FLAGS.img_width, FLAGS.img_height])
        projector.visualize_embeddings(summary_writer, config)
