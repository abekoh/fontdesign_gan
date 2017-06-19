from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras.utils.np_utils import to_categorical
import os
from PIL import Image

from models import Generator, Discriminator
from dataset import Dataset


class TrainingFontDesignGAN():

    def __init__(self):
        self._build_models()
        self.output_dir_path = 'output_train'
        if not os.path.exists(self.output_dir_path):
            os.mkdir(self.output_dir_path)

    def _build_models(self):

        self.discriminator = Discriminator()
        self.discriminator.compile(optimizer=Adam(lr=0.001, beta_1=0.5),
                                   loss=['binary_crossentropy', 'categorical_crossentropy'],
                                   loss_weights=[1., 1.])

        self.generator = Generator()
        self.generator.compile(optimizer=Adam(lr=0.001, beta_1=0.5),
                               loss='mean_absolute_error', loss_weights=[100.])

        self.discriminator.trainable = False
        self.generator_to_discriminator = Model(inputs=self.generator.input, outputs=self.discriminator(self.generator.output))
        self.generator_to_discriminator.compile(optimizer=Adam(lr=0.001, beta_1=0.5),
                                                loss=['binary_crossentropy', 'categorical_crossentropy'],
                                                loss_weights=[1., 1.])

        self.encoder = Model(inputs=self.generator.input[0], outputs=self.generator.get_layer('en_8').output)
        self.encoder.trainable = False
        self.generator_to_encoder = Model(inputs=self.generator.input, outputs=self.encoder(self.generator.output))
        self.generator_to_encoder.compile(optimizer=Adam(lr=0.001, beta_1=0.5),
                                          loss='mean_squared_error',
                                          loss_weights=[15.])

    def load_dataset(self, src_h5_path):
        dataset = Dataset()
        dataset.load_h5(src_h5_path)
        self.src_imgs, self.dst_imgs, self.char_ids, self.font_ids = dataset.get()
        self._shuffle_dataset()

    def _shuffle_dataset(self):
        combined = np.c_[self.dst_imgs.reshape(len(self.dst_imgs), -1), self.char_ids, self.font_ids]
        np.random.shuffle(combined)
        dst_imgs_n = self.dst_imgs.size // len(self.dst_imgs)
        self.dst_imgs = combined[:, :dst_imgs_n].reshape(self.dst_imgs.shape)
        self.char_ids = combined[:, dst_imgs_n:dst_imgs_n + 1].reshape(self.char_ids.shape)
        self.font_ids = combined[:, dst_imgs_n + 1:].reshape(self.font_ids.shape)

    def train(self):
        epoch_n = 100
        batch_size = 16
        embedding_n = 40
        batch_n = int(self.dst_imgs.shape[0] / batch_size)

        for epoch_i in range(epoch_n):

            for batch_i in range(batch_n):
                print('batch {} of {} / epoch {} of {}'.format(batch_i + 1, batch_n, epoch_i + 1, epoch_n))
                batched_dst_imgs = np.zeros((batch_size, 256, 256, 1), dtype=np.float32)
                batched_src_imgs = np.zeros((batch_size, 256, 256, 1), dtype=np.float32)
                batched_font_ids = np.zeros((batch_size), dtype=np.int32)
                for i, j in enumerate(range(batch_i * batch_size, (batch_i + 1) * batch_size)):
                    batched_dst_imgs[i, :, :, :] = self.dst_imgs[j]
                    batched_font_ids[i] = self.font_ids[j]
                    batched_src_imgs[i, :, :, :] = self.src_imgs[int(self.char_ids[j])]

                batched_generated_imgs = self.generator.predict_on_batch([batched_src_imgs, batched_font_ids])
                batched_src_imgs_encoded = self.encoder.predict_on_batch(batched_src_imgs)

                _, d_loss_real, real_category_loss = self.discriminator.train_on_batch(batched_dst_imgs, [np.ones((batch_size, 1), dtype=np.float32), to_categorical(batched_font_ids, embedding_n)])

                _, d_loss_fake, fake_category_loss_d = self.discriminator.train_on_batch(batched_generated_imgs, [np.zeros((batch_size, 1), dtype=np.float32), to_categorical(batched_font_ids, embedding_n)])

                _, cheat_loss, fake_category_loss_g = self.generator_to_discriminator.train_on_batch([batched_src_imgs, batched_font_ids], [np.ones((batch_size, 1), dtype=np.float32), to_categorical(batched_font_ids, embedding_n)])

                l1_loss = self.generator.train_on_batch([batched_src_imgs, batched_font_ids], batched_dst_imgs)

                const_loss = self.generator_to_encoder.train_on_batch([batched_src_imgs, batched_font_ids], batched_src_imgs_encoded)

                self._save_images(batched_dst_imgs, batched_generated_imgs, epoch_i, batch_i)

                category_loss = real_category_loss + fake_category_loss_d
                d_loss = d_loss_real + (real_category_loss + fake_category_loss_d) / 2.0
                g_loss = cheat_loss + l1_loss + fake_category_loss_g + const_loss

                log_format = 'd_loss: {}, g_loss: {}, category_loss: {}, cheat_loss: {}, const_loss: {}, l1_loss: {}, d_loss_real: {}, d_loss_fake: {}'
                print(log_format.format(d_loss, g_loss, category_loss, cheat_loss, const_loss, l1_loss, d_loss_real, d_loss_fake))

                if (batch_i % 100 == 0 and batch_i != 0) or batch_i == batch_n - 1:
                    self._save_model_weights(epoch_i, batch_i)

    def _save_images(self, dst_imgs, generated_imgs, epoch_i, batch_i):
        epoch_output_dir_path = os.path.join(self.output_dir_path, 'temp', '{}'.format(epoch_i))
        if not os.path.exists(epoch_output_dir_path):
            os.makedirs(epoch_output_dir_path)
        for img_i in range(dst_imgs.shape[0]):
            num_img = np.concatenate((dst_imgs[img_i], generated_imgs[img_i]), axis=1)
            num_img = num_img * 255
            num_img = np.reshape(num_img, (256, 512))
            pil_img = Image.fromarray(np.uint8(num_img))
            pil_img.save(os.path.join(epoch_output_dir_path, '{}_{}.png'.format(batch_i, img_i)))

    def _save_model_weights(self, epoch_i, batch_i):
        model_weights_dir_path = os.path.join(self.output_dir_path, 'model_weights')
        if not os.path.exists(model_weights_dir_path):
            os.makedirs(model_weights_dir_path)
        self.generator.save_weights(os.path.join(model_weights_dir_path, 'gen_{}_{}.h5'.format(epoch_i, batch_i)))
        self.discriminator.save_weights(os.path.join(model_weights_dir_path, 'dis_{}_{}.h5'.format(epoch_i, batch_i)))


if __name__ == '__main__':
    gan = TrainingFontDesignGAN()
    gan.load_dataset('./font_jp.h5')
    gan.train()
