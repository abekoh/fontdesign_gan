import numpy as np
import os
from PIL import Image
from keras.optimizers import Adam

from models import Generator
from dataset import Dataset


class GeneratingFontDesignGAN():

    def __init__(self, gen_h5_path):
        self._build_models(gen_h5_path)
        self.output_dir_path = 'output_generate'
        if not os.path.exists(self.output_dir_path):
            os.mkdir(self.output_dir_path)

    def _build_models(self, gen_h5_path):
        self.generator = Generator()
        self.generator.load_weights(gen_h5_path)
        self.generator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='mean_absolute_error')

    def load_dataset(self, src_h5_path):
        dataset = Dataset()
        dataset.load_h5(src_h5_path)
        self.src_imgs, _, _, _ = dataset.get()

    def generate(self, char_id, font_id):
        batched_src_imgs = np.zeros((1, 256, 256, 1), dtype=np.float32)
        batched_font_ids = np.zeros((1), dtype=np.int32)
        batched_src_imgs[0, :, :, :] = self.src_imgs[char_id]
        batched_font_ids[0] = font_id
        generated_imgs = self.generator.predict_on_batch([batched_src_imgs, batched_font_ids])
        self._save_images(generated_imgs[0])

    def _save_images(self, generated_img):
        num_img = generated_img
        num_img = num_img * 255
        num_img = np.reshape(num_img, (256, 256))
        pil_img = Image.fromarray(np.uint8(num_img))
        pil_img.save(os.path.join(self.output_dir_path, 'generated.png'))


if __name__ == '__main__':
    gan = GeneratingFontDesignGAN('./output/model_weights/gen_0_200.h5')
    gan.load_dataset('./font_jp.h5')
    gan.generate(3, 0)
