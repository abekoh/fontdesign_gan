import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from models import Discriminator
from utils import deconcat_imgs, remove_white_imgs

FLAGS = tf.app.flags.FLAGS


class VisualizationFontDesignGAN():

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._get_imgs()
        self._prepare_visualization()
        self._extract_intermediate()

    def _setup_dirs(self):
        self.src_log = os.path.join(FLAGS.src_gan, 'log')
        self.dst_visualization = os.path.join(FLAGS.src_gan, 'visualization')
        if not os.path.exists(self.dst_visualization):
            os.makedirs(self.dst_visualization)

    def _get_imgs(self):
        pil_img = Image.open(FLAGS.vis_imgs_path)
        np_img = np.asarray(pil_img)
        np_img = (np_img.astype(np.float32) / 127.5) - 1.
        deconcated_imgs = deconcat_imgs(np_img, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.src_imgs = remove_white_imgs(deconcated_imgs, 1.)
        self.batch_size = self.src_imgs.shape[0]

    def _prepare_visualization(self):
        self.imgs = tf.placeholder(tf.float32, (self.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim),
                                   name='imgs')
        discriminator = Discriminator(img_size=(FLAGS.img_width, FLAGS.img_height),
                                      img_dim=FLAGS.img_dim,
                                      layer_n=4,
                                      k_size=3,
                                      smallest_hidden_unit_n=64,
                                      is_bn=FLAGS.batchnorm)
        _, intermediate_xs = discriminator(self.src_imgs, is_train=False, is_intermediate=True)

        self.intermediate_tensors = list()
        with tf.variable_scope('intermediate'):
            for i, intermediate_x in enumerate(intermediate_xs):
                self.intermediate_tensors.append(tf.Variable(intermediate_x, name='layer{}'.format(i)))

        self.layer_n = len(intermediate_xs)

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
        self.sess.run(tf.global_variables_initializer(), feed_dict={self.imgs: self.src_imgs})

        d_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        pretrained_saver = tf.train.Saver(var_list=d_vars)
        checkpoint = tf.train.get_checkpoint_state(self.src_log)
        assert checkpoint, 'cannot get checkpoint: {}'.format(self.src_log)
        pretrained_saver.restore(self.sess, checkpoint.model_checkpoint_path)

        intermediate_vars = [var for var in tf.global_variables() if 'intermediate' in var.name]
        self.saver = tf.train.Saver(var_list=intermediate_vars)
        self.writer = tf.summary.FileWriter(self.dst_visualization)

    def _extract_intermediate(self):
        self.intermediate_tensors_nps = self.sess.run([self.intermediate_tensors[i] for i in range(self.layer_n)],
                                                      feed_dict={self.imgs: self.src_imgs})
        self.concated_intermediate_tensor_np = np.concatenate([v for v in self.intermediate_tensors_nps], axis=1)
        self.saver.save(self.sess, os.path.join(self.dst_visualization, 'result.ckpt'))

    def project_tensorboard(self):
        summary_writer = tf.summary.FileWriter(self.dst_visualization)
        config = projector.ProjectorConfig()
        config.model_checkpoint_path = os.path.realpath(os.path.join(self.dst_visualization, 'result.ckpt'))
        for i in range(self.layer_n):
            embedding_config = config.embeddings.add()
            embedding_config.tensor_name = 'intermediate/layer{}'.format(i)
            embedding_config.sprite.image_path = os.path.realpath(FLAGS.vis_imgs_path)
            embedding_config.sprite.single_image_dim.extend([FLAGS.img_width, FLAGS.img_height])
        projector.visualize_embeddings(summary_writer, config)

    def calc_tsne(self):
        font_n = self.batch_size // 26
        char_labels = [chr(i + 65) for i in np.tile(np.arange(0, 26), font_n).tolist()]
        style_labels = np.repeat(np.arange(0, font_n), 26)
        for layer_i in range(self.layer_n):
            reduced = TSNE(n_components=2, verbose=3, perplexity=40, n_iter=5000)
            reduced.fit_transform(self.intermediate_tensors_nps[layer_i])
            plt.figure(figsize=(16, 9))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=["w" for _ in char_labels])
            cmap = plt.get_cmap('hsv')
            for i in range(reduced.shape[0]):
                plt.text(reduced[i][0], reduced[i][1], char_labels[i],
                         fontdict={'size': 10, 'color': cmap(style_labels[i] / np.max(style_labels))})
            plt.savefig(os.path.join(self.dst_visualization, 'tsne_disc_layer{}.png'.format(layer_i)))
            plt.close()
