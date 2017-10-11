import os
import numpy as np
import json
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


class TrainingFontDesignGAN():

    def __init__(self, params, paths):

        self.params = params
        self.paths = paths

    def setup(self):
        self._set_dsts()
        self._build_models()
        self._load_dataset()
        self._prepare_training()
        self._save_params()

    def delete(self):
        # self.sess.close()
        tf.reset_default_graph()

    def _set_dsts(self):
        for path in self.paths.dst.__dict__.values():
            if not os.path.exists(path):
                os.makedirs(path)

    def _save_params(self):
        with open(os.path.join(self.paths.dst.root, 'params.txt'), 'w') as f:
            json.dump(self.params.to_dict(), f, indent=4)
        with open(os.path.join(self.paths.dst.root, 'paths.txt'), 'w') as f:
            json.dump(self.paths.to_dict(), f, indent=4)

    def _build_models(self):
        self.generator = models.Generator(img_size=self.params.img_size,
                                          img_dim=self.params.img_dim,
                                          z_size=self.params.z_size,
                                          layer_n=self.params.g.layer_n,
                                          k_size=self.params.g.k_size,
                                          smallest_hidden_unit_n=self.params.g.smallest_hidden_unit_n,
                                          is_bn=self.params.g.is_bn)
        self.discriminator = models.Discriminator(img_size=self.params.img_size,
                                                  img_dim=self.params.img_dim,
                                                  layer_n=self.params.d.layer_n,
                                                  k_size=self.params.d.k_size,
                                                  smallest_hidden_unit_n=self.params.d.smallest_hidden_unit_n,
                                                  is_bn=self.params.d.is_bn)
        if hasattr(self.params, 'c'):
            self.classifier = models.Classifier(img_size=self.params.img_size,
                                                img_dim=self.params.img_dim,
                                                k_size=self.params.c.k_size,
                                                class_n=26,
                                                smallest_unit_n=self.params.c.smallest_unit_n)

    def _load_dataset(self, is_shuffle=True):
        self.real_dataset = Dataset(self.paths.src.real_h5, 'r', img_size=self.params.img_size, img_dim=self.params.img_dim)
        self.real_dataset.set_load_data()
        if is_shuffle:
            self.real_dataset.shuffle()
        self.real_data_n = self.real_dataset.get_img_len()

    def _set_embeddings(self):
        self.font_z_size = int(self.params.z_size * self.params.font_embedding_rate)
        self.char_z_size = self.params.z_size - self.font_z_size

        self.font_embedding = np.random.uniform(-1, 1, (self.params.font_embedding_n, self.font_z_size))
        self.char_embedding = np.random.uniform(-1, 1, (self.params.char_embedding_n, self.char_z_size))

        with tf.variable_scope('embeddings'):
            tf.Variable(self.font_embedding, name='font_embedding')
            tf.Variable(self.char_embedding, name='char_embedding')

        embedding_h5file = h5py.File(os.path.join(self.paths.dst.root, 'embeddings.h5'), 'w')
        embedding_h5file.create_dataset('font_embedding', data=self.font_embedding)
        embedding_h5file.create_dataset('char_embedding', data=self.char_embedding)

    def _prepare_training(self):
        self._set_embeddings()

        self.real_imgs = tf.placeholder(tf.float32, (self.params.batch_size, self.params.img_size[0], self.params.img_size[1], self.params.img_dim), name='real_imgs')
        self.z = tf.placeholder(tf.float32, (self.params.batch_size, self.params.z_size), name='z')
        self.fake_imgs = self.generator.parallel(self.z)

        self.d_real = self.discriminator.parallel(self.real_imgs)
        self.d_fake = self.discriminator.parallel(self.fake_imgs, is_reuse=True)

        self.d_loss = - (tf.reduce_mean(self.d_real) - tf.reduce_mean(self.d_fake))
        self.g_loss = - tf.reduce_mean(self.d_fake)

        epsilon = tf.random_uniform((self.params.batch_size, 1, 1, 1), minval=0., maxval=1.)
        interp = self.real_imgs + epsilon * (self.fake_imgs - self.real_imgs)
        d_interp = self.discriminator.parallel(interp, is_reuse=True)
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

        if hasattr(self.params, 'c'):
            self.labels = tf.placeholder(tf.float32, (None, self.params.char_embedding_n))
            self.c_fake = self.params.c.penalty * self.classifier(self.fake_imgs)
            self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.c_fake))
            tf.summary.scalar('c_loss', self.c_loss)
            if self.params.c.opt == 'Adam':
                self.c_opt = tf.train.AdamOptimizer(learning_rate=self.params.c.lr, beta1=0.5, beta2=0.9).minimize(self.c_loss, var_list=g_vars)
            elif self.params.c.opt == 'RMSProp':
                self.c_opt = tf.train.RMSPropOptimizer(learning_rate=self.params.c.lr).minimize(self.c_loss, var_list=g_vars)
            elif self.params.c.opt == 'Adadelta':
                self.c_opt = tf.train.AdadeltaOptimizer().minimize(self.c_loss, var_list=g_vars)
            elif self.params.c.opt == 'SGD':
                self.c_opt = tf.train.GradientDescentOptimizer(learning_rate=self.params.c.lr).minimize(self.c_loss, var_list=g_vars)
            correct_pred = tf.equal(tf.argmax(self.c_fake, 1), tf.argmax(self.labels, 1))
            self.c_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('c_acc', self.c_acc)
            c_vars = [var for var in tf.global_variables() if 'classifier' in var.name]

        self.summary = tf.summary.merge_all()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver_pretrained = tf.train.Saver(var_list=c_vars)
        self.saver_pretrained.restore(self.sess, self.paths.src.classifier_ckpt)

        self.saver = tf.train.Saver()

        self.writer = tf.summary.FileWriter(self.paths.dst.log)

    def _get_z(self, font_ids=None, char_ids=None):
        if font_ids is not None:
            font_z = np.take(self.font_embedding, font_ids, axis=0)
        else:
            font_z = np.random.uniform(-1, 1, (self.params.batch_size, self.font_z_size))
        if char_ids is not None:
            char_z = np.take(self.char_embedding, char_ids, axis=0)
        else:
            char_z = np.random.uniform(-1, 1, (self.params.batch_size, self.char_z_size))
        z = np.concatenate((font_z, char_z), axis=1)
        return z

    def train(self):

        if self.params.is_run_tensorboard:
            self._run_tensorboard()

        batch_n = self.real_data_n // self.params.batch_size

        for epoch_i in tqdm(range(self.params.epoch_n)):

            for batch_i in tqdm(range(batch_n)):

                count_i = epoch_i * batch_n + batch_i

                for i in range(self.params.critic_n):

                    batched_real_imgs, _ = self.real_dataset.get_random(self.params.batch_size)
                    batched_z = self._get_z()

                    self.sess.run(self.d_opt, feed_dict={self.z: batched_z, self.real_imgs: batched_real_imgs})

                batched_z = self._get_z()

                self.sess.run(self.g_opt, feed_dict={self.z: batched_z})

                if hasattr(self.params, 'c'):
                    char_ids = np.random.randint(0, self.params.char_embedding_n, (self.params.batch_size), dtype=np.int32)
                    batched_z = self._get_z(char_ids=char_ids)
                    batched_labels = to_categorical(char_ids, self.params.char_embedding_n)
                    self.sess.run(self.c_opt, feed_dict={self.z: batched_z, self.labels: batched_labels})

                self.score, summary = self.sess.run([self.d_loss, self.summary],
                                                    feed_dict={self.z: batched_z,
                                                               self.labels: batched_labels,
                                                               self.real_imgs: batched_real_imgs})

                self.writer.add_summary(summary, count_i)

                # save images
                if (batch_i + 1) % self.params.save_imgs_interval == 0:
                    self.save_temp_imgs(os.path.join(self.paths.dst.sample, '{}_{}.png'.format(epoch_i + 1, batch_i + 1)))

            self.saver.save(self.sess, os.path.join(self.paths.dst.log, 'result_{}.ckpt'.format(epoch_i)))
            # self._visualize_embedding(epoch_i)

    def _run_tensorboard(self):
        Popen(['tensorboard', '--logdir', '{}'.format(os.path.realpath(self.paths.dst.log))], stdout=PIPE)

    def _generate_img(self, z, row_n, col_n):
        batched_generated_imgs = self.sess.run(self.fake_imgs, feed_dict={self.z: z})
        concated_img = concat_imgs(batched_generated_imgs, row_n, col_n)
        concated_img = (concated_img + 1.) * 127.5
        if self.params.img_dim == 1:
            concated_img = np.reshape(concated_img, (-1, col_n * self.params.img_size[0]))
        else:
            concated_img = np.reshape(concated_img, (-1, col_n * self.params.img_size[0], self.params.img_dim))
        return concated_img

    def _init_temp_imgs_inputs(self):
        temp_batched_src_fonts = np.concatenate((np.repeat(0, 26), np.random.randint(1, 256, (256 - 26))))
        temp_batched_src_chars = np.concatenate((np.arange(0, 26), np.repeat(0, 128 - 26), np.random.randint(1, 26, (128))))
        # temp_batched_src_fonts = np.random.randint(0, self.params.font_embedding_n, (self.params.temp_imgs_n), dtype=np.int32)
        # temp_batched_src_chars = np.random.randint(0, self.params.char_embedding_n, (self.params.temp_imgs_n), dtype=np.int32)
        self.temp_batched_z = self._get_z(font_ids=temp_batched_src_fonts, char_ids=temp_batched_src_chars)

    def save_temp_imgs(self, filepath):
        if not hasattr(self, 'temp_batched_z'):
            self._init_temp_imgs_inputs()
        row_n = self.params.temp_imgs_n // self.params.temp_col_n
        concated_img = self._generate_img(self.temp_batched_z, row_n, self.params.temp_col_n)
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(filepath)

    def _save_weights(self, epoch_i, batch_i):
        self.generator.save_weights(os.path.join(self.paths.dst.model_weights, 'gen_{}_{}.h5'.format(epoch_i + 1, batch_i + 1)))
        self.discriminator.save_weights(os.path.join(self.paths.dst.model_weights, 'dis_{}_{}.h5'.format(epoch_i + 1, batch_i + 1)))

    def _init_visualize_imgs_inputs(self):
        font_vis_font_ids = np.arange(0, self.params.font_embedding_n, dtype=np.int32)
        font_vis_char_ids = np.repeat(np.array([0], dtype=np.int32), self.params.font_embedding_n)
        self.font_vis_z = self._get_z(font_vis_font_ids, font_vis_char_ids)

        char_vis_font_ids = np.repeat(np.array([0], dtype=np.int32), self.params.char_embedding_n)
        char_vis_char_ids = np.arange(0, self.params.char_embedding_n, dtype=np.int32)
        self.char_vis_z = self._get_z(char_vis_font_ids, char_vis_char_ids)

    def _visualize_embedding(self, epoch_i):
        if not hasattr(self, 'font_vis_z'):
            self._init_visualize_imgs_inputs()
        font_vis_img_path = os.path.realpath(os.path.join(self.paths.dst.log, 'font_vis_{}.png'.format(epoch_i)))
        char_vis_img_path = os.path.realpath(os.path.join(self.paths.dst.log, 'char_vis_{}.png'.format(epoch_i)))

        font_vis_img = self._generate_img(self.font_vis_z, 16, 16)
        font_vis_img = Image.fromarray(np.uint8(font_vis_img))
        font_vis_img.save(font_vis_img_path)

        char_vis_img = self._generate_img(self.char_vis_z, 6, 6)
        char_vis_img = Image.fromarray(np.uint8(char_vis_img))
        char_vis_img.save(char_vis_img_path)

        summary_writer = tf.summary.FileWriter(self.paths.dst.log)
        config = projector.ProjectorConfig()
        font_embedding = config.embeddings.add()
        font_embedding.tensor_name = self.font_embedding_tf.name
        font_embedding.sprite.image_path = font_vis_img_path
        font_embedding.sprite.single_image_dim.extend([64, 64])
        char_embedding = config.embeddings.add()
        char_embedding.tensor_name = self.char_embedding_tf.name
        char_embedding.sprite.image_path = char_vis_img_path
        char_embedding.sprite.single_image_dim.extend([64, 64])
        projector.visualize_embeddings(summary_writer, config)

    def get_score(self):
        return self.score
