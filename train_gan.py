import os
import numpy as np
import h5py
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
        os.mkdir(FLAGS.dst_root)
        os.mkdir(FLAGS.dst_log)
        os.mkdir(FLAGS.dst_samples)

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

    def _set_embeddings(self):
        self.font_z_size = int(FLAGS.z_size * FLAGS.font_embedding_rate)
        self.char_z_size = FLAGS.z_size - self.font_z_size

        self.font_embedding = np.random.uniform(-1, 1, (FLAGS.font_embedding_n, self.font_z_size))
        self.char_embedding = np.random.uniform(-1, 1, (FLAGS.char_embedding_n, self.char_z_size))

        with tf.variable_scope('embeddings'):
            tf.Variable(self.font_embedding, name='font_embedding')
            tf.Variable(self.char_embedding, name='char_embedding')

        embedding_h5file = h5py.File(os.path.join(FLAGS.dst_root, 'embeddings.h5'), 'w')
        embedding_h5file.create_dataset('font_embedding', data=self.font_embedding)
        embedding_h5file.create_dataset('char_embedding', data=self.char_embedding)

    def _prepare_training(self):
        self._set_embeddings()

        self.real_imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='real_imgs')
        self.z = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.z_size), name='z')
        self.fake_imgs = self.generator(self.z)

        self.d_real = self.discriminator(self.real_imgs)
        self.d_fake = self.discriminator(self.fake_imgs, is_reuse=True)

        self.d_loss = - (tf.reduce_mean(self.d_real) - tf.reduce_mean(self.d_fake))
        self.g_loss = - tf.reduce_mean(self.d_fake)

        epsilon = tf.random_uniform((FLAGS.batch_size, 1, 1, 1), minval=0., maxval=1.)
        interp = self.real_imgs + epsilon * (self.fake_imgs - self.real_imgs)
        d_interp = self.discriminator(interp, is_reuse=True)
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
            self.c_fake = FLAGS.c_penalty * self.classifier(self.fake_imgs)
            self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.c_fake))
            tf.summary.scalar('c_loss', self.c_loss)
            self.c_opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.c_lr).minimize(self.c_loss, var_list=g_vars)
            correct_pred = tf.equal(tf.argmax(self.c_fake, 1), tf.argmax(self.labels, 1))
            self.c_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('c_acc', self.c_acc)
            c_vars = [var for var in tf.global_variables() if 'classifier' in var.name]

        self.summary = tf.summary.merge_all()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver_pretrained = tf.train.Saver(var_list=c_vars)
        self.saver_pretrained.restore(self.sess, FLAGS.src_classifier_ckpt)

        self.saver = tf.train.Saver()

        self.writer = tf.summary.FileWriter(FLAGS.dst_log)

    def _get_z(self, font_ids=None, char_ids=None):
        if font_ids is not None:
            font_z = np.take(self.font_embedding, font_ids, axis=0)
        else:
            font_z = np.random.uniform(-1, 1, (FLAGS.batch_size, self.font_z_size))
        if char_ids is not None:
            char_z = np.take(self.char_embedding, char_ids, axis=0)
        else:
            char_z = np.random.uniform(-1, 1, (FLAGS.batch_size, self.char_z_size))
        z = np.concatenate((font_z, char_z), axis=1)
        return z

    def train(self):

        if FLAGS.is_run_tensorboard:
            self._run_tensorboard()

        batch_n = self.real_data_n // FLAGS.batch_size

        for epoch_i in tqdm(range(FLAGS.epoch_n)):

            for batch_i in tqdm(range(batch_n)):

                count_i = epoch_i * batch_n + batch_i

                for i in range(FLAGS.critic_n):

                    batched_real_imgs, _ = self.real_dataset.get_random(FLAGS.batch_size)
                    batched_z = self._get_z()

                    self.sess.run(self.d_opt, feed_dict={self.z: batched_z, self.real_imgs: batched_real_imgs})

                batched_z = self._get_z()

                self.sess.run(self.g_opt, feed_dict={self.z: batched_z})

                if FLAGS.c_penalty > 0.:
                    char_ids = np.random.randint(0, FLAGS.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)
                    batched_z = self._get_z(char_ids=char_ids)
                    batched_labels = to_categorical(char_ids, FLAGS.char_embedding_n)
                    self.sess.run(self.c_opt, feed_dict={self.z: batched_z, self.labels: batched_labels})

                self.score, summary = self.sess.run([self.d_loss, self.summary],
                                                    feed_dict={self.z: batched_z,
                                                               self.labels: batched_labels,
                                                               self.real_imgs: batched_real_imgs})

                self.writer.add_summary(summary, count_i)

                # save images
                if (batch_i + 1) % FLAGS.save_imgs_interval == 0:
                    self.save_temp_imgs(os.path.join(FLAGS.dst_samples, '{}_{}.png'.format(epoch_i + 1, batch_i + 1)))

            self.saver.save(self.sess, os.path.join(FLAGS.dst_log, 'result_{}.ckpt'.format(epoch_i)))
            self._visualize_embedding(epoch_i)

    def _run_tensorboard(self):
        Popen(['tensorboard', '--logdir', '{}'.format(os.path.realpath(FLAGS.dst_log))], stdout=PIPE)

    def _generate_img(self, z, row_n, col_n):
        batched_generated_imgs = self.sess.run(self.fake_imgs, feed_dict={self.z: z})
        concated_img = concat_imgs(batched_generated_imgs, row_n, col_n)
        concated_img = (concated_img + 1.) * 127.5
        if FLAGS.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, col_n * FLAGS.img_height))
        else:
            concated_img = np.reshape(concated_img, (-1, col_n * FLAGS.img_height, FLAGS.img_dim))
        return concated_img

    def _init_temp_imgs_inputs(self):
        temp_batched_src_fonts = np.concatenate((np.repeat(0, 26), np.random.randint(1, 256, (256 - 26))))
        temp_batched_src_chars = np.concatenate((np.arange(0, 26), np.repeat(0, 128 - 26), np.random.randint(1, 26, (128))))
        # temp_batched_src_fonts = np.random.randint(0, FLAGS.font_embedding_n, (FLAGS.temp_imgs_n), dtype=np.int32)
        # temp_batched_src_chars = np.random.randint(0, FLAGS.char_embedding_n, (FLAGS.temp_imgs_n), dtype=np.int32)
        self.temp_batched_z = self._get_z(font_ids=temp_batched_src_fonts, char_ids=temp_batched_src_chars)

    def save_temp_imgs(self, filepath):
        if not hasattr(self, 'temp_batched_z'):
            self._init_temp_imgs_inputs()
        concated_img = self._generate_img(self.temp_batched_z, 16, 16)
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(filepath)

    def _save_weights(self, epoch_i, batch_i):
        self.generator.save_weights(os.path.join(FLAGS.dst_model_weights, 'gen_{}_{}.h5'.format(epoch_i + 1, batch_i + 1)))
        self.discriminator.save_weights(os.path.join(FLAGS.dst_model_weights, 'dis_{}_{}.h5'.format(epoch_i + 1, batch_i + 1)))

    def _init_visualize_imgs_inputs(self):
        font_vis_font_ids = np.arange(0, FLAGS.font_embedding_n, dtype=np.int32)
        font_vis_char_ids = np.repeat(np.array([0], dtype=np.int32), FLAGS.font_embedding_n)
        self.font_vis_z = self._get_z(font_vis_font_ids, font_vis_char_ids)

    def _visualize_embedding(self, epoch_i):
        if not hasattr(self, 'font_vis_z'):
            self._init_visualize_imgs_inputs()
        font_vis_img_path = os.path.realpath(os.path.join(FLAGS.dst_log, 'font_vis_{}.png'.format(epoch_i)))

        font_vis_img = self._generate_img(self.font_vis_z, 16, 16)
        font_vis_img = Image.fromarray(np.uint8(font_vis_img))
        font_vis_img.save(font_vis_img_path)

        summary_writer = tf.summary.FileWriter(FLAGS.dst_log)
        config = projector.ProjectorConfig()
        font_embedding = config.embeddings.add()
        font_embedding.tensor_name = 'embeddings/font_embedding'
        font_embedding.sprite.image_path = font_vis_img_path
        font_embedding.sprite.single_image_dim.extend([FLAGS.img_width, FLAGS.img_height])
        projector.visualize_embeddings(summary_writer, config)

    def get_score(self):
        return self.score
