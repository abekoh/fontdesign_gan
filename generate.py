import numpy as np
import os
from PIL import Image
from keras.optimizers import Adam
from params import Params

import models
from dataset import Dataset


class GeneratingFontDesignGAN():

    def __init__(self, params, paths):
        self.params = params
        self.paths = paths
        if not os.path.exists(self.paths.dst):
            os.makedirs(self.paths.dst)
        self._build_model()
        self._load_dataset()

    def _build_model(self):
        if self.params.g_arch == 'pix2pix':
            self.generator = models.GeneratorPix2Pix(img_size=self.params.img_size,
                                                     img_dim=self.params.img_dim,
                                                     font_embedding_n=self.params.font_embedding_n)
        if self.params.g_arch == 'dcgan':
            self.generator = models.GeneratorDCGAN(img_size=self.params.img_size,
                                                   img_dim=self.params.img_dim,
                                                   font_embedding_n=self.params.font_embedding_n,
                                                   char_embedding_n=self.params.char_embedding_n)
        self.generator.load_weights(self.paths.src_model_weight_h5)
        self.generator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='mean_absolute_error')

    def _load_dataset(self):
        self.dataset = Dataset(self.paths.src_fonts_h5, 'r', img_size=self.params.img_size)
        self.dataset.set_load_data()

    def generate(self, char, font_id, filename='generated.png'):
        batched_src_fonts = np.array([font_id], dtype=np.int32)
        batched_src_chars, _ = self.dataset.get_selected([char])
        generated_imgs = self.generator.predict_on_batch([batched_src_chars, batched_src_fonts])
        self._save_image(generated_imgs[0], filename)

    def _save_image(self, generated_img, filename):
        num_img = generated_img
        num_img = num_img * 255
        num_img = np.reshape(num_img, (256, 256))
        pil_img = Image.fromarray(np.uint8(num_img))
        pil_img.save(os.path.join(self.paths.dst, filename))


if __name__ == '__main__':
    params = Params({
        'img_size': (256, 256),
        'img_dim': 1,
        'font_embedding_n': 5,
        'char_embedding_n': 26,
        'g_arch': 'pix2pix'
    })

    paths = Params({
        'src_model_weight_h5': 'output/2017-08-01 19:05:03.788443+09:00/model_weights/gen_45.h5',
        'src_fonts_h5': 'src/arial.h5',
        'dst': 'output/generate'
    })

    gan = GeneratingFontDesignGAN(params, paths)
    for i in range(5):
        gan.generate('A', i, 'A_{}.png'.format(i))
