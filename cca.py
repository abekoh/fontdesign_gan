import os

import tensorflow as tf
import numpy as np
import h5py
from sklearn.cross_decomposition import CCA
from matplotlib import pyplot as plt

from dataset import Dataset

FLAGS = tf.app.flags.FLAGS


class RunningCCA():

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._load_data()

    def _setup_dirs(self):
        self.dst_cca = os.path.join(FLAGS.gan_dir, 'cca')
        if not os.path.exists(self.dst_cca):
            os.mkdir(self.dst_cca)

    def _load_data(self):
        self.eigen_font_z_h5 = h5py.File(os.path.join(FLAGS.gan_dir, 'random_walking', 'eigen_font_z.h5'), 'r')
        self.generated_dataset = Dataset(FLAGS.generated_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.generated_dataset.set_load_data()

    def run_CCA(self, char):
        char_img_n = self.generated_dataset.get_data_n_by_labels([char])
        eigen_font_z = self.eigen_font_z_h5['eigen_font_z/params'].value
        imgs = self.generated_dataset.get_batch_by_labels(0, char_img_n, [char])
        imgs = np.reshape(imgs, (char_img_n, -1))
        print(eigen_font_z.shape)
        print(imgs.shape)
        cca = CCA(n_components=1)
        cca.fit(eigen_font_z, imgs)
        eigen_font_z_c, imgs_c = cca.transform(eigen_font_z, imgs)
        print(eigen_font_z_c.shape)
        print(imgs_c.shape)

        corrcoef = np.corrcoef(eigen_font_z_c.T, imgs_c.T)
        print(corrcoef)
