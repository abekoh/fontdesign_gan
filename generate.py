import numpy as np
import os
from PIL import Image
import h5py
import tensorflow as tf
from keras import backend as K

import models
from utils import concat_imgs


class GeneratingFontDesignGAN():

    def __init__(self, params, paths):
        self.sess = tf.Session()
        K.set_session(self.sess)

        self.params = params
        self.paths = paths
        if not os.path.exists(self.paths.dst):
            os.makedirs(self.paths.dst)
        self._build_model()
        self._prepare_generating()

    def _build_model(self):
        if self.params.g_arch == 'dcgan':
            self.generator = models.GeneratorDCGAN_NoEmbedding(img_size=self.params.img_size,
                                                               img_dim=self.params.img_dim,
                                                               layer_n=4,
                                                               smallest_hidden_unit_n=128)
        self.generator.load_weights(self.paths.src_model_weight_h5)

    def _prepare_generating(self):
        embedding_h5file = h5py.File(self.paths.src_embedding_h5, 'r')
        self.font_embedding = embedding_h5file['font_embedding'].value
        self.char_embedding = embedding_h5file['char_embedding'].value

        self.z = tf.placeholder(tf.float32, (None, 100), name='z')
        self.generated_imgs = self.generator(self.z)

    def _get_embedded(self, font_ids, char_ids):
        if len(font_ids) % self.params.batch_size != 0:
            pad_n = self.params.batch_size - len(font_ids) % self.params.batch_size
            font_ids = np.concatenate((font_ids, np.random.randint(0, self.params.font_embedding_n, pad_n)), axis=0)
            char_ids = np.concatenate((char_ids, np.random.randint(0, self.params.char_embedding_n, pad_n)), axis=0)
        font_embedded = np.take(self.font_embedding, font_ids, axis=0)
        char_embedded = np.take(self.char_embedding, char_ids, axis=0)
        z = np.concatenate((font_embedded, char_embedded), axis=1)
        return z

    def generate(self, font_ids, char_ids, col_n=10, filename='generated.png'):
        batched_z = self._get_embedded(font_ids, char_ids)
        batched_generated_imgs = self.sess.run(self.generated_imgs, feed_dict={self.z: batched_z,
                                                                               K.learning_phase(): 0})
        if font_ids.shape[0] > col_n:
            row_n = font_ids.shape[0] // col_n + 1
        else:
            col_n = font_ids.shape[0]
            row_n = 1
        concated_img = concat_imgs(batched_generated_imgs, row_n, col_n)
        concated_img = (concated_img + 1.) * 127.5
        concated_img = np.reshape(concated_img, (-1, col_n * self.params.img_size[0]))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(os.path.join(self.paths.dst, filename))
