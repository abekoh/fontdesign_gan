from keras.models import Model
from models import Generator, Discriminator
from keras.optimizers import Adam
from PIL import Image
import numpy as np
from glob import glob
from keras.utils.np_utils import to_categorical
from keras.utils import Progbar
import os

CAPS = [chr(i) for i in range(65, 65 + 26)]


class FontDesignGAN():

    def __init__(self):
        self._build_models()
        self.output_dir_path = 'output'
        if not os.path.exists(self.output_dir_path):
            os.mkdir(self.output_dir_path)

    def _build_models(self):

        self.discriminator = Discriminator()
        self.discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                                   loss=['binary_crossentropy', 'categorical_crossentropy'],
                                   loss_weights=[1., 0.5])

        self.generator = Generator()
        self.generator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                               loss='mean_absolute_error', loss_weights=[100.])

        self.discriminator.trainable = False
        self.generator_to_discriminator = Model(inputs=self.generator.input, outputs=self.discriminator(self.generator.output))
        self.generator_to_discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                                                loss=['binary_crossentropy', 'categorical_crossentropy'],
                                                loss_weights=[1., 1.])

        self.encoder = Model(inputs=self.generator.input[0], outputs=self.generator.get_layer('en_8').output)
        self.encoder.trainable = False
        self.generator_to_encoder = Model(inputs=self.generator.input, outputs=self.encoder(self.generator.output))
        self.generator_to_encoder.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                                          loss='mean_squared_error',
                                          loss_weight=[1.])

    def _get_dir_paths(self, dir_path):
        dir_paths = []
        for filename in glob(os.path.join(dir_path, '*')):
            if os.path.isdir(filename):
                dir_paths.append(filename)
        dir_paths.sort()
        return dir_paths

    def _load_font_imgs(self, font_dir_path, is_src, font_id=0):
        for char_id, c in enumerate(CAPS):
            img_pil = Image.open(os.path.join(font_dir_path, '{}.png'.format(c)))
            img_np = np.asarray(img_pil)
            img_np = img_np.astype(np.float32) / 255.0
            img_np = img_np[np.newaxis, :, :, np.newaxis]
            if is_src:
                self.src_imgs = np.append(self.src_imgs, img_np, axis=0)
            else:
                self.dst_imgs = np.append(self.dst_imgs, img_np, axis=0)
                self.char_ids = np.append(self.char_ids, np.array([char_id]))
                self.font_ids = np.append(self.font_ids, np.array([font_id]))

    def load_dataset(self, fonts_dir_path, src_font_name):
        self.src_imgs = np.empty((0, 256, 256, 1), np.float32)
        self.dst_imgs = np.empty((0, 256, 256, 1), np.float32)
        self.char_ids = np.array([])
        self.font_ids = np.array([])
        font_dir_paths = self._get_dir_paths(fonts_dir_path)
        src_font_dir_path = os.path.join(fonts_dir_path, src_font_name)
        font_id = 0
        for font_dir_path in font_dir_paths:
            print('loading: {}'.format(font_dir_path))
            if font_dir_path == src_font_dir_path:
                self._load_font_imgs(font_dir_path, True)
            else:
                self._load_font_imgs(font_dir_path, False, font_id=font_id)
                font_id += 1

    def save_images(self, dst_imgs, generated_imgs, epoch_i, batch_i):
        epoch_output_dir_path = os.path.join(self.output_dir_path, '{}'.format(epoch_i))
        if not os.path.exists(epoch_output_dir_path):
            os.mkdir(epoch_output_dir_path)
        for img_i in range(dst_imgs.shape[0]):
            num_img = np.concatenate((dst_imgs[img_i], generated_imgs[img_i]), axis=1)
            num_img = num_img * 255
            num_img = np.reshape(num_img, (256, 512))
            pil_img = Image.fromarray(np.uint8(num_img))
            pil_img.save(os.path.join(epoch_output_dir_path, '{}_{}.png'.format(batch_i, img_i)))

    def train(self):
        epoch_n = 1000
        batch_size = 16
        embedding_n = 160
        batch_n = int(self.dst_imgs.shape[0] / batch_size)

        for epoch_i in range(epoch_n):
            print('epoch {} of {}'.format(epoch_i + 1, epoch_n))
            progress_bar = Progbar(target=batch_n)

            for batch_i in range(batch_n):
                progress_bar.update(batch_i)
                batched_dst_imgs = np.zeros((batch_size, 256, 256, 1), dtype=np.float32)
                batched_src_imgs = np.zeros((batch_size, 256, 256, 1), dtype=np.float32)
                batched_font_ids = np.zeros((batch_size), dtype=np.int32)
                for i, j in enumerate(range(batch_i * batch_size, (batch_i + 1) * batch_size)):
                    batched_dst_imgs[i, :, :, :] = self.dst_imgs[j]
                    batched_src_imgs[i, :, :, :] = self.src_imgs[j % 26]
                    batched_font_ids[i] = self.font_ids[j]

                batched_generated_imgs = self.generator.predict_on_batch([batched_src_imgs, batched_font_ids])
                batched_src_imgs_encoded = self.encoder.predict_on_batch(batched_src_imgs)

                self.discriminator.train_on_batch(batched_dst_imgs, [np.zeros(batch_size, dtype=np.int32), to_categorical(batched_font_ids, embedding_n)])
                self.discriminator.train_on_batch(batched_generated_imgs, [np.ones(batch_size, dtype=np.int32), to_categorical(batched_font_ids, embedding_n)])

                self.generator_to_discriminator.train_on_batch([batched_src_imgs, batched_font_ids], [np.zeros(batch_size, dtype=np.int32), to_categorical(batched_font_ids, embedding_n)])

                self.generator.train_on_batch([batched_src_imgs, batched_font_ids], batched_dst_imgs)

                self.generator_to_encoder.train_on_batch([batched_src_imgs, batched_font_ids], batched_src_imgs_encoded)

                self.save_images(batched_dst_imgs, batched_generated_imgs, epoch_i, batch_i)


if __name__ == '__main__':
    gan = FontDesignGAN()
    gan.load_dataset('../../font_dataset/font_160/', 'Atkins-Regular')
    gan.train()
