import os
import json
import time
import math
from subprocess import Popen, PIPE

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import Dataset
from models import Generator, Discriminator, Classifier
from ops import average_gradients
from utils import concat_imgs

FLAGS = tf.app.flags.FLAGS
ALPHABET_CAPS = list(chr(i) for i in range(65, 65 + 26))
HIRAGANA_SEION = list('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐゑをん')


class TrainingFontDesignGAN():

    def __init__(self):
        global FLAGS
        self._check_params()
        self._setup_dirs()
        self._save_flags()
        self._prepare_training()
        self._load_dataset()

    def reset(self):
        '''
        Reset graph
        '''
        tf.reset_default_graph()

    def _check_params(self):
        '''
        Check parameters
        '''
        assert FLAGS.batch_size >= FLAGS.font_embedding_n, 'batch_size must be greater equal than font_embedding_n'

    def _setup_dirs(self):
        '''
        Setup output directories
        '''
        if not os.path.exists(FLAGS.dst_gan):
            os.makedirs(FLAGS.dst_gan)
        self.dst_log = os.path.join(FLAGS.dst_gan, 'log')
        self.dst_log_fontemb = os.path.join(self.dst_log, 'font_embedding')
        self.dst_samples = os.path.join(FLAGS.dst_gan, 'sample')
        if not os.path.exists(self.dst_log_fontemb):
            os.makedirs(self.dst_log_fontemb)
        if not os.path.exists(self.dst_samples):
            os.mkdir(self.dst_samples)

    def _save_flags(self):
        '''
        Save FLAGS as JSON
        '''
        with open(os.path.join(self.dst_log, 'flags.json'), 'w') as f:
            json.dump(FLAGS.__dict__['__flags'], f, indent=4)

    def _load_dataset(self):
        '''
        Load dataset
        '''
        self.real_dataset = Dataset(FLAGS.font_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.real_dataset.set_load_data()
        self.real_dataset.shuffle()

    def _prepare_training(self):
        '''
        Prepare Training
        '''
        self.font_z_size = int(FLAGS.z_size * FLAGS.font_embedding_rate)
        self.char_z_size = FLAGS.z_size - self.font_z_size
        self.gpu_n = len(FLAGS.gpu_ids.split(','))
        self.embedding_chars = list()
        if 'caps' in FLAGS.embedding_chars_type:
            self.embedding_chars.extend(ALPHABET_CAPS)
        if 'hiragana' in FLAGS.embedding_chars_type:
            self.embedding_chars.extend(HIRAGANA_SEION)
        assert self.embedding_chars != [], 'embedding_chars is empty'
        self.char_embedding_n = len(self.embedding_chars)

        with tf.device('/cpu:0'):
            # Set embeddings from uniform distribution
            font_embedding_np = np.random.uniform(-1, 1, (FLAGS.font_embedding_n, self.font_z_size)).astype(np.float32)
            char_embedding_np = np.random.uniform(-1, 1, (self.char_embedding_n, self.char_z_size)).astype(np.float32)
            with tf.variable_scope('embeddings'):
                self.font_embedding = tf.Variable(font_embedding_np, name='font_embedding')
                self.char_embedding = tf.Variable(char_embedding_np, name='char_embedding')

            self.font_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='font_ids')
            self.char_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='char_ids')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
            self.real_imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='real_imgs')
            self.labels = tf.placeholder(tf.float32, (FLAGS.batch_size, self.char_embedding_n), name='labels')

            d_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)
            g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)
            c_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.c_lr, beta1=0.5, beta2=0.9)

        # Initialize lists for multi gpu
        fake_imgs = [0] * self.gpu_n
        d_loss = [0] * self.gpu_n
        g_loss = [0] * self.gpu_n
        c_loss = [0] * self.gpu_n
        c_acc = [0] * self.gpu_n

        d_grads = [0] * self.gpu_n
        g_grads = [0] * self.gpu_n
        c_grads = [0] * self.gpu_n

        divided_batch_size = FLAGS.batch_size // self.gpu_n
        is_not_first = False

        # Build graph
        for i in range(self.gpu_n):
            batch_start = i * divided_batch_size
            batch_end = (i + 1) * divided_batch_size
            with tf.device('/gpu:{}'.format(i)):
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
                classifier = Classifier(img_size=(FLAGS.img_width, FLAGS.img_height),
                                        img_dim=FLAGS.img_dim,
                                        k_size=3,
                                        class_n=26,
                                        smallest_unit_n=64)

                # If sum of (font/char)_ids is less than -1, z is generated from uniform distribution
                font_z = tf.cond(tf.less(tf.reduce_sum(self.font_ids[batch_start:batch_end]), 0),
                                 lambda: tf.random_uniform((divided_batch_size, self.font_z_size), -1, 1),
                                 lambda: tf.nn.embedding_lookup(self.font_embedding, self.font_ids[batch_start:batch_end]))
                char_z = tf.cond(tf.less(tf.reduce_sum(self.char_ids[batch_start:batch_end]), 0),
                                 lambda: tf.random_uniform((divided_batch_size, self.char_z_size), -1, 1),
                                 lambda: tf.nn.embedding_lookup(self.char_embedding, self.char_ids[batch_start:batch_end]))
                z = tf.concat([font_z, char_z], axis=1)

                # Generate fake images
                fake_imgs[i] = generator(z, is_reuse=is_not_first, is_train=self.is_train)

                # Calculate loss
                d_real = discriminator(self.real_imgs[batch_start:batch_end], is_reuse=is_not_first, is_train=self.is_train)
                d_fake = discriminator(fake_imgs[i], is_reuse=True, is_train=self.is_train)
                d_loss[i] = - (tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))
                g_loss[i] = - tf.reduce_mean(d_fake)

                # Calculate gradient Penalty
                epsilon = tf.random_uniform((divided_batch_size, 1, 1, 1), minval=0., maxval=1.)
                interp = self.real_imgs[batch_start:batch_end] + epsilon * (fake_imgs[i] - self.real_imgs[batch_start:batch_end])
                d_interp = discriminator(interp, is_reuse=True, is_train=self.is_train)
                grads = tf.gradients(d_interp, [interp])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[-1]))
                grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                d_loss[i] += 10 * grad_penalty

                # Get trainable variables
                d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
                g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

                d_grads[i] = d_opt.compute_gradients(d_loss[i], var_list=d_vars)
                g_grads[i] = g_opt.compute_gradients(g_loss[i], var_list=g_vars)

                # for training with Classifier
                if FLAGS.c_penalty != 0.:
                    c_fake = FLAGS.c_penalty * classifier(fake_imgs[i], is_reuse=is_not_first, is_train=self.is_train)
                    c_loss[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels[batch_start:batch_end], logits=c_fake))
                    correct_pred = tf.equal(tf.argmax(c_fake, 1), tf.argmax(self.labels[batch_start:batch_end], 1))
                    c_acc[i] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                    c_grads[i] = c_opt.compute_gradients(c_loss[i], var_list=g_vars)
            is_not_first = True

        with tf.device('/cpu:0'):
            self.fake_imgs = tf.concat(fake_imgs, axis=0)
            avg_d_grads = average_gradients(d_grads)
            avg_g_grads = average_gradients(g_grads)
            self.d_train = d_opt.apply_gradients(avg_d_grads)
            self.g_train = g_opt.apply_gradients(avg_g_grads)
            if FLAGS.c_penalty != 0:
                avg_c_grads = average_gradients(c_grads)
                self.c_train = c_opt.apply_gradients(avg_c_grads)

        # Calculate summary for tensorboard
        tf.summary.scalar('d_loss', -(sum(d_loss) / len(d_loss)))
        tf.summary.scalar('g_loss', -(sum(g_loss) / len(g_loss)))
        if FLAGS.c_penalty != 0:
            tf.summary.scalar('c_loss', sum(c_loss) / len(c_loss))
            tf.summary.scalar('c_acc', sum(c_acc) / len(c_acc))
        self.summary = tf.summary.merge_all()

        # Setup session
        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=FLAGS.ckpt_keep_n,
                                    keep_checkpoint_every_n_hours=FLAGS.keep_ckpt_hour)

        # If checkpoint is found, restart training
        checkpoint = tf.train.get_checkpoint_state(self.dst_log)
        if checkpoint:
            saver_resume = tf.train.Saver()
            saver_resume.restore(self.sess, checkpoint.model_checkpoint_path)
            self.epoch_start = int(checkpoint.model_checkpoint_path.split('-')[-1]) + 1
            print('restore ckpt')
        else:
            self.sess.run(tf.global_variables_initializer())
            # Load Classifier's weight
            if FLAGS.c_penalty != 0:
                src_log = os.path.join(FLAGS.src_classifier, 'log')
                classifier_checkpoint = tf.train.get_checkpoint_state(src_log)
                assert classifier_checkpoint, 'not found classifier\'s checkpoint: {}'.format(src_log)
                c_vars = [var for var in tf.global_variables() if 'classifier' in var.name]
                saver_pretrained = tf.train.Saver(var_list=c_vars)
                saver_pretrained.restore(self.sess, classifier_checkpoint.model_checkpoint_path)
            self.epoch_start = 0

        # Setup writer for tensorboard
        self.writer = tf.summary.FileWriter(self.dst_log)

    def _get_ids(self, font_selector=None, char_selector=None):
        '''
        Get embedding ids
        '''
        if type(font_selector) == int:
            font_ids = np.repeat(0, font_selector, (FLAGS.batch_size), dtype=np.int32)
        elif font_selector == 'random':
            font_ids = np.random.randint(0, FLAGS.font_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        else:
            # All ids are -1 -> z is generated from uniform distribution when calculate graph
            font_ids = np.ones(FLAGS.batch_size) * -1
        if type(char_selector) == str and len(char_selector) == 1:
            char_ids = np.repeat(self.real_dataset.get_ids_from_labels(char_selector)[0], FLAGS.batch_size).astype(np.int32)
        elif char_selector == 'random':
            char_ids = np.random.randint(0, self.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        else:
            # All ids are -1 -> z is generated from uniform distribution when calculate graph
            char_ids = np.ones(FLAGS.batch_size) * -1
        return font_ids, char_ids

    def train(self):
        '''
        Train GAN
        '''
        # Start tensorboard
        if FLAGS.run_tensorboard:
            self._run_tensorboard()

        for epoch_i in tqdm(range(self.epoch_start, FLAGS.gan_epoch_n), initial=self.epoch_start, total=FLAGS.gan_epoch_n):
            for embedding_char in self.embedding_chars:
                # Approximate wasserstein distance
                for critic_i in range(FLAGS.critic_n):
                    real_imgs = self.real_dataset.get_random_by_labels(FLAGS.batch_size, [embedding_char])
                    font_ids, char_ids = self._get_ids(None, embedding_char)
                    self.sess.run(self.d_train, feed_dict={self.font_ids: font_ids,
                                                           self.char_ids: char_ids,
                                                           self.real_imgs: real_imgs,
                                                           self.is_train: True})

                # Minimize wasserstein distance
                font_ids, char_ids = self._get_ids(None, embedding_char)
                self.sess.run(self.g_train, feed_dict={self.font_ids: font_ids,
                                                       self.char_ids: char_ids,
                                                       self.is_train: True})

                # Maximize character likelihood
                if FLAGS.c_penalty != 0.:
                    labels = np.eye(self.char_embedding_n)[char_ids]
                    self.sess.run(self.c_train, feed_dict={self.font_ids: font_ids,
                                                           self.char_ids: char_ids,
                                                           self.labels: labels,
                                                           self.is_train: True})

            # Calculate losses for tensorboard
            real_imgs = self.real_dataset.get_random(FLAGS.batch_size, is_label=False)
            font_ids, char_ids = self._get_ids(None, 'random')
            feed = {self.font_ids: font_ids, self.char_ids: char_ids, self.real_imgs: real_imgs, self.is_train: True}
            if FLAGS.c_penalty != 0.:
                labels = np.eye(self.char_embedding_n)[char_ids]
                feed[self.labels] = labels
            summary = self.sess.run(self.summary, feed_dict=feed)

            self.writer.add_summary(summary, epoch_i)

            # Save model weights
            self.saver.save(self.sess, os.path.join(self.dst_log, 'result.ckpt'), global_step=epoch_i + 1)

            # Save sample images
            if (epoch_i + 1) % FLAGS.sample_imgs_interval == 0:
                self._save_sample_imgs(epoch_i + 1)

            # Save images for tensorboard projector
            if (epoch_i + 1) % FLAGS.embedding_imgs_interval == 0:
                self._save_embedding_imgs(epoch_i + 1)

    def _run_tensorboard(self):
        '''
        Run tensorboard
        '''
        Popen(['tensorboard', '--logdir', '{}'.format(os.path.realpath(self.dst_log)), '--port', '{}'.format(FLAGS.tensorboard_port)], stdout=PIPE)
        time.sleep(1)

    def _generate_img(self, font_ids, char_ids, row_n, col_n):
        '''
        Generate image
        '''
        feed = {self.font_ids: font_ids, self.char_ids: char_ids, self.is_train: False}
        generated_imgs = self.sess.run(self.fake_imgs, feed_dict=feed)
        combined_img = concat_imgs(generated_imgs, row_n, col_n)
        combined_img = (combined_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            combined_img = np.reshape(combined_img, (-1, col_n * FLAGS.img_height))
        else:
            combined_img = np.reshape(combined_img, (-1, col_n * FLAGS.img_height, FLAGS.img_dim))
        return Image.fromarray(np.uint8(combined_img))

    def _init_save_imgs_edge_n(self):
        '''
        Initialize save images' num of edge
        '''
        self.save_imgs_edge_n = math.ceil(math.sqrt(FLAGS.batch_size))

    def _init_sample_imgs_inputs(self):
        '''
        Initialize inputs for generating sample images
        '''
        if not hasattr(self, 'save_imgs_edge_n'):
            self._init_save_imgs_edge_n()
        self.sample_font_ids = np.random.randint(0, FLAGS.font_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        self.sample_char_ids = np.random.randint(0, self.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)

    def _save_sample_imgs(self, epoch_i):
        '''
        Save sample images
        '''
        if not hasattr(self, 'sample_font_ids'):
            self._init_sample_imgs_inputs()
        concated_img = self._generate_img(self.sample_font_ids, self.sample_char_ids,
                                          self.save_imgs_edge_n, self.save_imgs_edge_n)
        concated_img.save(os.path.join(self.dst_samples, '{}.png'.format(epoch_i)))

    def _init_embedding_imgs_inputs(self):
        '''
        Initialize inputs for generating embedding images
        '''
        if not hasattr(self, 'save_imgs_edge_n'):
            self._init_save_imgs_edge_n()
        self.embedding_font_ids = np.arange(0, FLAGS.font_embedding_n, dtype=np.int32)
        self.embedding_char_ids = np.repeat(np.array([0], dtype=np.int32), FLAGS.font_embedding_n)
        if FLAGS.batch_size > FLAGS.font_embedding_n:
            expand_ids = np.repeat(np.array([0], dtype=np.int32), FLAGS.batch_size - FLAGS.font_embedding_n)
            self.embedding_font_ids = np.concatenate((self.embedding_font_ids, expand_ids), axis=0)
            self.embedding_char_ids = np.concatenate((self.embedding_char_ids, expand_ids), axis=0)

    def _save_embedding_imgs(self, epoch_i):
        '''
        Save embedding images
        '''
        if not hasattr(self, 'embedding_font_ids'):
            self._init_embedding_imgs_inputs()
        embedding_img_path = os.path.realpath(os.path.join(self.dst_log_fontemb, '{}.png'.format(epoch_i)))
        embedding_img = self._generate_img(self.embedding_font_ids, self.embedding_char_ids,
                                           self.save_imgs_edge_n, self.save_imgs_edge_n)
        embedding_img.save(embedding_img_path)
        summary_writer = tf.summary.FileWriter(self.dst_log)
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = 'embeddings/font_embedding'
        embedding_config.sprite.image_path = embedding_img_path
        embedding_config.sprite.single_image_dim.extend([FLAGS.img_width, FLAGS.img_height])
        projector.visualize_embeddings(summary_writer, config)
