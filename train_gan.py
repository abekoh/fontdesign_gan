import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from subprocess import Popen, PIPE

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from models import Generator, Discriminator, Classifier
from dataset import Dataset
from utils import concat_imgs

FLAGS = tf.app.flags.FLAGS


class TrainingFontDesignGAN():

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._prepare_training()
        self._load_dataset()

    def reset(self):
        tf.reset_default_graph()

    def _setup_dirs(self):
        if not os.path.exists(FLAGS.dst_gan):
            os.makedirs(FLAGS.dst_gan)
        self.dst_log = os.path.join(FLAGS.dst_gan, 'log')
        self.dst_log_fontemb = os.path.join(self.dst_log, 'font_embedding')
        self.dst_samples = os.path.join(FLAGS.dst_gan, 'sample')
        if not os.path.exists(self.dst_log_fontemb):
            os.makedirs(self.dst_log_fontemb)
        if not os.path.exists(self.dst_samples):
            os.mkdir(self.dst_samples)

    def _load_dataset(self):
        self.real_dataset = Dataset(FLAGS.font_h5, 'r', img_size=(FLAGS.img_width, FLAGS.img_height), img_dim=FLAGS.img_dim, is_mem=True)
        if FLAGS.chars != '':
            ids = [int(s) for s in FLAGS.chars.replace(' ', '').split(',')]
            self.real_dataset.set_load_data(ids=ids)
        else:
            self.real_dataset.set_load_data()
        self.real_dataset.shuffle()
        self.real_data_n = self.real_dataset.get_img_len()

    def _prepare_training(self):

        self.font_z_size = int(FLAGS.z_size * FLAGS.font_embedding_rate)
        self.char_z_size = FLAGS.z_size - self.font_z_size
        # FLAGS.batch_size = FLAGS.batch_size // FLAGS.gpu_n

        font_embedding_np = np.random.uniform(-1, 1, (FLAGS.font_embedding_n, self.font_z_size)).astype(np.float32)
        char_embedding_np = np.random.uniform(-1, 1, (FLAGS.char_embedding_n, self.char_z_size)).astype(np.float32)

        with tf.device('/gpu:{}'.format(FLAGS.gpu_n - 1)):
            with tf.variable_scope('embeddings'):
                self.font_embedding = tf.Variable(font_embedding_np, name='font_embedding')
                self.char_embedding = tf.Variable(char_embedding_np, name='char_embedding')

            self.font_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='font_ids')
            self.char_ids = tf.placeholder(tf.int32, (FLAGS.batch_size,), name='char_ids')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
            self.real_imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='real_imgs')
            self.labels = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.char_embedding_n), name='labels')

            d_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)
            g_opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9)
            c_opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.c_lr)

        fake_imgs = [0] * FLAGS.gpu_n
        d_loss = [0] * FLAGS.gpu_n
        g_loss = [0] * FLAGS.gpu_n
        c_loss = [0] * FLAGS.gpu_n
        c_acc = [0] * FLAGS.gpu_n

        d_grads = [0] * FLAGS.gpu_n
        g_grads = [0] * FLAGS.gpu_n
        c_grads = [0] * FLAGS.gpu_n

        d_reuse = [True for _ in range(FLAGS.gpu_n)]
        g_reuse = [True for _ in range(FLAGS.gpu_n)]
        c_reuse = [True for _ in range(FLAGS.gpu_n)]
        d_reuse[0] = False
        g_reuse[0] = False
        c_reuse[0] = False

        divided_batch_size = FLAGS.batch_size // FLAGS.gpu_n

        for i in range(FLAGS.gpu_n):
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

                font_z = tf.cond(tf.less(tf.reduce_sum(self.font_ids[batch_start:batch_end]), 0),
                                 lambda: tf.random_uniform((divided_batch_size, self.font_z_size), -1, 1),
                                 lambda: tf.nn.embedding_lookup(self.font_embedding, self.font_ids[batch_start:batch_end]))
                char_z = tf.cond(tf.less(tf.reduce_sum(self.char_ids[batch_start:batch_end]), 0),
                                 lambda: tf.random_uniform((divided_batch_size, self.char_z_size), -1, 1),
                                 lambda: tf.nn.embedding_lookup(self.char_embedding, self.char_ids[batch_start:batch_end]))
                z = tf.concat([font_z, char_z], axis=1)

                fake_imgs[i] = generator(z, is_reuse=g_reuse[i], is_train=self.is_train)

                d_real = discriminator(self.real_imgs[batch_start:batch_end], is_reuse=d_reuse[i], is_train=self.is_train)
                d_fake = discriminator(fake_imgs[i], is_reuse=True, is_train=self.is_train)

                d_loss[i] = - (tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))
                g_loss[i] = - tf.reduce_mean(d_fake)

                epsilon = tf.random_uniform((divided_batch_size, 1, 1, 1), minval=0., maxval=1.)
                interp = self.real_imgs[batch_start:batch_end] + epsilon * (fake_imgs[i] - self.real_imgs[batch_start:batch_end])
                d_interp = discriminator(interp, is_reuse=True, is_train=self.is_train)
                grads = tf.gradients(d_interp, [interp])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[-1]))
                grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                d_loss[i] += 10 * grad_penalty

                d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
                g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]

                d_grads[i] = d_opt.compute_gradients(d_loss[i], var_list=d_vars)
                g_grads[i] = g_opt.compute_gradients(g_loss[i], var_list=g_vars)

                if FLAGS.c_penalty != 0.:
                    c_fake = FLAGS.c_penalty * classifier(fake_imgs[i], is_reuse=c_reuse[i], is_train=self.is_train)
                    c_loss[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels[batch_start:batch_end], logits=c_fake))
                    correct_pred = tf.equal(tf.argmax(c_fake, 1), tf.argmax(self.labels[batch_start:batch_end], 1))
                    c_acc[i] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                    c_grads[i] = c_opt.compute_gradients(c_loss[i], var_list=g_vars)

        with tf.device('/gpu:{}'.format(FLAGS.gpu_n - 1)):
            sum_d_grads = self._average_gradients(d_grads)
            sum_g_grads = self._average_gradients(g_grads)
            self.d_train = d_opt.apply_gradients(sum_d_grads)
            self.g_train = g_opt.apply_gradients(sum_g_grads)
            if FLAGS.c_penalty != 0:
                sum_c_grads = self._average_gradients(c_grads)
                self.c_train = c_opt.apply_gradients(sum_c_grads)

        self.fake_imgs = tf.concat(fake_imgs, axis=0)

        tf.summary.scalar('d_loss', sum(d_loss))
        tf.summary.scalar('g_loss', sum(g_loss))
        if FLAGS.c_penalty != 0:
            tf.summary.scalar('c_loss', sum(c_loss))
            tf.summary.scalar('c_acc', sum(c_acc) / len(c_acc))
        self.summary = tf.summary.merge_all()

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver(max_to_keep=FLAGS.ckpt_keep_n,
                                    keep_checkpoint_every_n_hours=FLAGS.keep_ckpt_hour)

        checkpoint = tf.train.get_checkpoint_state(self.dst_log)
        if checkpoint:
            saver_resume = tf.train.Saver()
            saver_resume.restore(self.sess, checkpoint.model_checkpoint_path)
            self.epoch_start = int(checkpoint.model_checkpoint_path.split('-')[-1]) + 1
            print('restore ckpt')
        else:
            self.sess.run(tf.global_variables_initializer())
            if FLAGS.c_penalty != 0:
                src_log = os.path.join(FLAGS.src_classifier, 'log')
                classifier_checkpoint = tf.train.get_checkpoint_state(src_log)
                assert classifier_checkpoint, 'not found classifier\'s checkpoint: {}'.format(src_log)
                c_vars = [var for var in tf.global_variables() if 'classifier' in var.name]
                saver_pretrained = tf.train.Saver(var_list=c_vars)
                saver_pretrained.restore(self.sess, classifier_checkpoint.model_checkpoint_path)
            self.epoch_start = 0

        self.writer = tf.summary.FileWriter(self.dst_log)

    def _average_gradients(self, tower_grads):
        averaged_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, axis=0)
                grads.append(expanded_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            averaged_grads.append(grad_and_var)
        return averaged_grads

    def _get_ids(self, embedding_font_ids=None, embedding_char_ids=None):
        if type(embedding_font_ids) == int:
            font_ids = np.repeat(0, embedding_font_ids, (FLAGS.batch_size), dtype=np.int32)
        elif embedding_font_ids == 'random':
            font_ids = np.random.randint(0, FLAGS.font_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        else:
            font_ids = np.ones(FLAGS.batch_size) * -1
        if type(embedding_char_ids) == int:
            char_ids = np.repeat(embedding_char_ids, FLAGS.batch_size).astype(np.int32)
        elif embedding_char_ids == 'random':
            char_ids = np.random.randint(0, FLAGS.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        else:
            char_ids = np.ones(FLAGS.batch_size) * -1
        return font_ids, char_ids

    def train(self):

        if FLAGS.run_tensorboard:
            self._run_tensorboard()

        for epoch_i in tqdm(range(self.epoch_start, FLAGS.gan_epoch_n), initial=self.epoch_start, total=FLAGS.gan_epoch_n):

            for char_i in range(FLAGS.char_embedding_n):

                for critic_i in range(FLAGS.critic_n):

                    real_imgs = self.real_dataset.get_random_by_ids(FLAGS.batch_size, [char_i], is_label=False)
                    font_ids, char_ids = self._get_ids(None, char_i)

                    feed = {self.font_ids: font_ids, self.char_ids: char_ids, self.real_imgs: real_imgs, self.is_train: True}
                    self.sess.run(self.d_train, feed_dict=feed)

                font_ids, char_ids = self._get_ids(None, char_i)

                feed = {self.font_ids: font_ids, self.char_ids: char_ids, self.is_train: True}
                self.sess.run(self.g_train, feed_dict=feed)

                if FLAGS.c_penalty != 0.:
                    font_ids, char_ids = self._get_ids(None, 'random')
                    labels = [np.eye(FLAGS.char_embedding_n)[char_ids[i]] for i in range(FLAGS.gpu_n)]
                    feed = {self.font_ids: font_ids, self.char_ids: char_ids, self.labels: labels, self.is_train: True}
                    self.sess.run(self.c_train, feed_dict=feed)

                feed = {self.font_ids: font_ids, self.char_ids: char_ids, self.real_imgs: real_imgs, self.is_train: True}
                if FLAGS.c_penalty != 0.:
                    feed[self.labels] = labels
                summary = self.sess.run(self.summary, feed_dict=feed)

                self.writer.add_summary(summary, epoch_i)

            # save images
            if (epoch_i + 1) % FLAGS.sample_imgs_interval == 0:
                self._save_sample_imgs(epoch_i + 1)

            self.saver.save(self.sess, os.path.join(self.dst_log, 'result.ckpt'), global_step=epoch_i + 1)

            if (epoch_i + 1) % FLAGS.embedding_interval == 0:
                self._save_embedding_imgs(epoch_i + 1)

    def _run_tensorboard(self):
        Popen(['tensorboard', '--logdir', '{}'.format(os.path.realpath(self.dst_log)), '--port', '{}'.format(FLAGS.tensorboard_port)], stdout=PIPE)

    def _generate_img(self, font_ids, char_ids, row_n, col_n):
        feed = {self.font_ids: font_ids, self.char_ids: char_ids, self.is_train: False}
        generated_imgs = self.sess.run(self.fake_imgs, feed_dict=feed)
        combined_img = concat_imgs(generated_imgs, row_n, col_n)
        combined_img = (combined_img + 1.) * 127.5
        if FLAGS.flip:
            combined_img = combined_img * -1.
        if FLAGS.img_dim == 1:
            combined_img = np.reshape(combined_img, (-1, col_n * FLAGS.img_height))
        else:
            combined_img = np.reshape(combined_img, (-1, col_n * FLAGS.img_height, FLAGS.img_dim))
        return Image.fromarray(np.uint8(combined_img))

    def _init_sample_imgs_inputs(self):
        self.sample_font_ids = np.random.randint(0, FLAGS.font_embedding_n, (FLAGS.batch_size), dtype=np.int32)
        self.sample_char_ids = np.random.randint(0, FLAGS.char_embedding_n, (FLAGS.batch_size), dtype=np.int32)

    def _save_sample_imgs(self, epoch_i):
        if not hasattr(self, 'sample_font_ids'):
            self._init_sample_imgs_inputs()
        concated_img = self._generate_img(self.sample_font_ids, self.sample_char_ids,
                                          FLAGS.save_imgs_col_n, FLAGS.save_imgs_col_n)
        concated_img.save(os.path.join(self.dst_samples, '{}.png'.format(epoch_i)))

    def _init_embedding_imgs_inputs(self):
        self.embedding_font_ids = np.arange(0, FLAGS.font_embedding_n, dtype=np.int32)
        self.embedding_char_ids = np.repeat(np.array([0], dtype=np.int32), FLAGS.font_embedding_n)

    def _save_embedding_imgs(self, epoch_i):
        if not hasattr(self, 'embedding_font_ids'):
            self._init_embedding_imgs_inputs()
        embedding_img_path = os.path.realpath(os.path.join(self.dst_log_fontemb, '{}.png'.format(epoch_i)))
        embedding_img = self._generate_img(self.embedding_font_ids, self.embedding_char_ids,
                                           FLAGS.save_imgs_col_n, FLAGS.save_imgs_col_n)
        embedding_img.save(embedding_img_path)

        summary_writer = tf.summary.FileWriter(self.dst_log)
        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = 'embeddings/font_embedding'
        embedding_config.sprite.image_path = embedding_img_path
        embedding_config.sprite.single_image_dim.extend([FLAGS.img_width, FLAGS.img_height])
        projector.visualize_embeddings(summary_writer, config)
