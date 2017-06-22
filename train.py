import os
import numpy as np
import h5py
from PIL import Image
import plotly.offline as offline
import plotly.graph_objs as go

from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from models import Generator, Discriminator
from dataset import Dataset


class TrainingFontDesignGAN():

    def __init__(self, output_dir_path='output_train'):
        self._set_outputs(output_root_dir_path=output_dir_path)

    def _set_outputs(self, output_root_dir_path):
        self.output_root_dir_path = output_root_dir_path
        if not os.path.exists(self.output_root_dir_path):
            os.mkdir(self.output_root_dir_path)
        output_dir_names = ['generated_imgs', 'model_weights', 'losses']
        self.output_dir_paths = {}
        for output_dir_name in output_dir_names:
            self.output_dir_paths[output_dir_name] = os.path.join(output_root_dir_path, output_dir_name)
            if not os.path.exists(self.output_dir_paths[output_dir_name]):
                os.mkdir(self.output_dir_paths[output_dir_name])

    def build_models(self, img_dim=1, embedding_n=40, lr=0.0002, beta_1=0.5):
        self.img_dim = img_dim
        self.embedding_n = embedding_n

        self.discriminator = Discriminator(img_dim=self.img_dim, embedding_n=self.embedding_n)
        self.discriminator.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                                   loss=['binary_crossentropy', 'categorical_crossentropy'],
                                   loss_weights=[1., 0.5])

        self.generator = Generator(img_dim=self.img_dim, embedding_n=self.embedding_n)
        self.generator.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                               loss='mean_absolute_error', loss_weights=[100.])

        self.discriminator.trainable = False
        self.generator_to_discriminator = Model(inputs=self.generator.input, outputs=self.discriminator(self.generator.output))
        self.generator_to_discriminator.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                                                loss=['binary_crossentropy', 'categorical_crossentropy'],
                                                loss_weights=[1., 1.])

        self.encoder = Model(inputs=self.generator.input[0], outputs=self.generator.get_layer('en_last').output)
        self.encoder.trainable = False
        self.generator_to_encoder = Model(inputs=self.generator.input, outputs=self.encoder(self.generator.output))
        self.generator_to_encoder.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                                          loss='mean_squared_error',
                                          loss_weights=[15.])

    def load_dataset(self, src_h5_path, is_shuffle=True):
        dataset = Dataset()
        dataset.load_h5(src_h5_path)
        self.src_imgs, self.dst_imgs, self.char_ids, self.font_ids = dataset.get()
        if is_shuffle:
            self._shuffle_dataset()

    def _shuffle_dataset(self):
        combined = np.c_[self.dst_imgs.reshape(len(self.dst_imgs), -1), self.char_ids, self.font_ids]
        np.random.shuffle(combined)
        dst_imgs_n = self.dst_imgs.size // len(self.dst_imgs)
        self.dst_imgs = combined[:, :dst_imgs_n].reshape(self.dst_imgs.shape)
        self.char_ids = combined[:, dst_imgs_n:dst_imgs_n + 1].reshape(self.char_ids.shape)
        self.font_ids = combined[:, dst_imgs_n + 1:].reshape(self.font_ids.shape)

    def train(self, epoch_n=30, batch_size=16):
        self._init_losses_progress()

        batch_n = int(self.dst_imgs.shape[0] / batch_size)

        for epoch_i in range(epoch_n):
            for batch_i in range(batch_n):
                print('batch {} of {} / epoch {} of {}'.format(batch_i + 1, batch_n, epoch_i + 1, epoch_n))

                batched_src_imgs = np.zeros((batch_size, 256, 256, 1), dtype=np.float32)
                batched_dst_imgs = np.zeros((batch_size, 256, 256, 1), dtype=np.float32)
                batched_font_ids = np.zeros((batch_size), dtype=np.int32)
                for i, j in enumerate(range(batch_i * batch_size, (batch_i + 1) * batch_size)):
                    batched_src_imgs[i, :, :, :] = self.src_imgs[int(self.char_ids[j])]
                    batched_dst_imgs[i, :, :, :] = self.dst_imgs[j]
                    batched_font_ids[i] = self.font_ids[j]

                batched_generated_imgs = self.generator.predict_on_batch([batched_src_imgs, batched_font_ids])
                batched_src_imgs_encoded = self.encoder.predict_on_batch(batched_src_imgs)

                losses = {}

                _, losses['d_real_bin'], losses['d_real_cat'] = \
                    self.discriminator.train_on_batch(
                        batched_dst_imgs,
                        [np.ones((batch_size, 1), dtype=np.float32), to_categorical(batched_font_ids, self.embedding_n)])

                _, losses['d_fake_bin'], losses['d_fake_cat'] = \
                    self.discriminator.train_on_batch(
                        batched_generated_imgs,
                        [np.zeros((batch_size, 1), dtype=np.float32), to_categorical(batched_font_ids, self.embedding_n)])

                _, losses['g_fake_bin'], losses['g_fake_cat'] = \
                    self.generator_to_discriminator.train_on_batch(
                        [batched_src_imgs, batched_font_ids],
                        [np.ones((batch_size, 1), dtype=np.float32), to_categorical(batched_font_ids, self.embedding_n)])

                losses['g_l1'] = \
                    self.generator.train_on_batch(
                        [batched_src_imgs, batched_font_ids],
                        batched_dst_imgs)

                losses['g_const'] = \
                    self.generator_to_encoder.train_on_batch(
                        [batched_src_imgs, batched_font_ids],
                        batched_src_imgs_encoded)

                losses['d_cat'] = (losses['d_real_cat'] + losses['d_fake_cat']) * 0.5
                losses['d_bin'] = losses['d_real_bin'] + losses['d_fake_bin']
                losses['d'] = losses['d_bin'] + losses['d_cat']
                losses['g'] = losses['g_fake_bin'] + losses['g_fake_cat'] + losses['g_l1'] + losses['g_const']

                self._update_losses_progress(losses, epoch_i, batch_i, batch_n)
                self._save_losses_progress_html()

                if batch_i % 50 == 0 or batch_i == batch_n - 1:
                    self._save_images(batched_dst_imgs, batched_generated_imgs, epoch_i, batch_i)

            self._save_model_weights(epoch_i)
        self._save_losses_progress_h5()

    def _init_losses_progress(self):
        self.x_time = np.array([])
        self.y_losses = {}

    def _update_losses_progress(self, losses, epoch_i, batch_i, batch_n):
        self.x_time = np.append(self.x_time, np.array([epoch_i * batch_n + batch_i]))
        if self.y_losses == {}:
            for k, v in losses.items():
                self.y_losses[k] = np.array([v], dtype=np.float32)
        else:
            for k, v in losses.items():
                self.y_losses[k] = np.append(self.y_losses[k], np.array([v]))

    def _save_losses_progress_html(self):
        graphs = []
        for k, v in self.y_losses.items():
            graph = go.Scatter(x=self.x_time, y=v, mode='lines', name=k)
            offline.plot([graph], filename=os.path.join(self.output_dir_paths['losses'], '{}.html'.format(k)), auto_open=False)
            graphs.append(graph)
        layout = {'yaxis': {'range': [0, 100.]}}
        fig = {'data': graphs, 'layout': layout}
        offline.plot(fig, filename=os.path.join(self.output_dir_paths['losses'], 'all.html'), auto_open=False)

    def _save_losses_progress_h5(self):
        h5file = h5py.File(os.path.join(self.output_dir_paths['losses'], 'all.h5'))
        h5file.create_dataset('x_time', data=self.x_time)
        h5file.create_dataset('y_losses', data=self.y_losses)
        h5file.flush()
        h5file.close()

    def _save_images(self, dst_imgs, generated_imgs, epoch_i, batch_i):
        concatenated_num_img = np.empty((0, 512))
        for img_i in range(dst_imgs.shape[0]):
            num_img = np.concatenate((dst_imgs[img_i], generated_imgs[img_i]), axis=1)
            num_img = num_img * 255
            num_img = np.reshape(num_img, (256, 512))
            concatenated_num_img = np.concatenate((concatenated_num_img, num_img), axis=0)
            concatenated_num_img = np.reshape(concatenated_num_img, (-1, 512))
        pil_img = Image.fromarray(np.uint8(concatenated_num_img))
        pil_img.save(os.path.join(self.output_dir_paths['generated_imgs'], '{}_{}.png'.format(epoch_i, batch_i)))

    def _save_model_weights(self, epoch_i):
        self.generator.save_weights(os.path.join(self.output_dir_paths['model_weights'], 'gen_{}.h5'.format(epoch_i)))
        self.discriminator.save_weights(os.path.join(self.output_dir_paths['model_weights'], 'dis_{}.h5'.format(epoch_i)))


if __name__ == '__main__':
    gan = TrainingFontDesignGAN()
    gan.build_models()
    gan.load_dataset('./font_jp.h5')
    gan.train()
