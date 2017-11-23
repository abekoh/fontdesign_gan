import os

import tensorflow as tf
import numpy as np
from PIL import Image

from models import Discriminator
from utils import deconcat_imgs, remove_white_imgs

FLAGS = tf.app.flags.FLAGS


class VisualizationFontDesignGAN():

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        # self._prepare_visualization()
        self._get_imgs()

    def _setup_dirs(self):
        self.src_log = os.path.join(FLAGS.src_gan, 'log')
        self.dst_visualization = os.path.join(FLAGS.src_gan, 'visualization')
        if not os.path.exists(self.dst_visualization):
            os.makedirs(self.dst_visualization)

    def _prepare_visualization(self):
        self.imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='imgs')

        discriminator = Discriminator(img_size=(FLAGS.img_width, FLAGS.img_height),
                                      img_dim=FLAGS.img_dim,
                                      layer_n=4,
                                      k_size=3,
                                      smallest_hidden_unit_n=64,
                                      is_bn=FLAGS.batchnorm)
        self.discriminated = discriminator(self.imgs, is_train=False)

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

        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.src_log)
        assert checkpoint, 'cannot get checkpoint: {}'.format(self.src_log)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def _get_imgs(self):
        pil_img = Image.open(FLAGS.vis_imgs_path)
        np_img = np.asarray(pil_img)
        np_img = (np_img.astype(np.float32) / 127.5) - 1.
        deconcated_imgs = deconcat_imgs(np_img, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        deconcated_imgs = remove_white_imgs(deconcated_imgs, 1.)
        print(deconcated_imgs.shape)
        deconcated_imgs = (deconcated_imgs + 1.) * 127.5
        for i in range(1024):
            img = Image.fromarray(np.uint8(deconcated_imgs[i]))
            img.save(os.path.join(self.dst_visualization, '{}.png'.format(i)))

    def visualize(self, imgs):
        self.sess.run(self.discriminated, feed_dict={self.imgs: imgs})
