import os
import numpy as np
import h5py
from PIL import Image
import plotly.offline as offline
import plotly.graph_objs as go

from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Progbar, to_categorical

from models import Generator, Discriminator, Classifier
from dataset import Dataset
from ops import mean_only, wasserstein_distance


class TrainingFontDesignGAN():

    def __init__(self, dst_dir_path='output_gan'):
        self._set_dsts(dst_root_dir_path=dst_dir_path)

    def _set_dsts(self, dst_root_dir_path):
        self.dst_root_dir_path = dst_root_dir_path
        if not os.path.exists(self.dst_root_dir_path):
            os.mkdir(self.dst_root_dir_path)
        dst_dir_names = ['generated_imgs', 'model_weights', 'losses']
        self.dst_dir_paths = {}
        for dst_dir_name in dst_dir_names:
            self.dst_dir_paths[dst_dir_name] = os.path.join(dst_root_dir_path, dst_dir_name)
            if not os.path.exists(self.dst_dir_paths[dst_dir_name]):
                os.mkdir(self.dst_dir_paths[dst_dir_name])

    def build_models(self, classifier_h5_path, img_dim=1, embedding_n=40, lr=0.0002, beta_1=0.5,
                     loss_weights={'d': [1.], 'g2d': [1.], 'g2e': [15.], 'g2c': [0.1]}):
        self.img_dim = img_dim
        self.embedding_n = embedding_n

        self.discriminator = Discriminator(img_dim=self.img_dim, embedding_n=self.embedding_n)
        self.discriminator.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                                   loss=wasserstein_distance,
                                   loss_weights=loss_weights['d'])

        self.generator = Generator(img_dim=self.img_dim, embedding_n=self.embedding_n)
        # self.generator.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
        #                        loss='mean_absolute_error', loss_weights=[100.])

        self.discriminator.trainable = False
        self.generator_to_discriminator = Model(inputs=self.generator.input, outputs=self.discriminator(self.generator.output))
        self.generator_to_discriminator.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                                                loss=mean_only,
                                                loss_weights=loss_weights['g2d'])

        self.encoder = Model(inputs=self.generator.input[0], outputs=self.generator.get_layer('en_last').output)
        self.encoder.trainable = False
        self.generator_to_encoder = Model(inputs=self.generator.input, outputs=self.encoder(self.generator.output))
        self.generator_to_encoder.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                                          loss='mean_squared_error',
                                          loss_weights=loss_weights['g2e'])

        self.classifier = Classifier(img_dim=img_dim, class_n=26)
        self.classifier.load_weights(classifier_h5_path)
        self.classifier.trainable = False
        self.generator_to_classifier = Model(inputs=self.generator.input, outputs=self.classifier(self.generator.output))
        self.generator_to_classifier.compile(optimizer=Adam(lr=lr, beta_1=beta_1),
                                             loss='categorical_crossentropy',
                                             loss_weights=loss_weights['g2c'])

    def load_dataset(self, src_real_h5_path, src_src_h5_path, is_shuffle=True):
        self.real_dataset = Dataset(src_real_h5_path, 'r')
        self.real_dataset.set_load_data()
        if is_shuffle:
            self.real_dataset.shuffle()
        self.real_data_n = self.real_dataset.get_img_len()
        self.src_dataset = Dataset(src_src_h5_path, 'r')
        self.src_dataset.set_load_data()

    def train(self, epoch_n=30, batch_size=16, save_imgs_interval=10):
        self._init_losses_progress()

        batch_n = self.real_data_n // batch_size

        for epoch_i in range(epoch_n):

            progbar = Progbar(batch_n)

            for batch_i in range(batch_n):

                progbar.update(batch_i + 1)

                batched_font_ids = np.random.randint(0, 40, batch_size, dtype=np.int32)
                batched_src_imgs, batched_src_labels = self.src_dataset.get_random(batch_size)
                batched_real_imgs, _ = self.real_dataset.get_batch(batch_i, batch_size)

                batched_categorical_src_labels = self._labels_to_categorical(batched_src_labels)

                batched_fake_imgs = self.generator.predict_on_batch([batched_src_imgs, batched_font_ids])
                batched_src_imgs_encoded = self.encoder.predict_on_batch(batched_src_imgs)

                losses = dict()

                losses['d_real_bin'] = \
                    self.discriminator.train_on_batch(
                        batched_real_imgs,
                        np.ones((batch_size, 1), dtype=np.float32))

                losses['d_fake_bin'] = \
                    self.discriminator.train_on_batch(
                        batched_fake_imgs,
                        np.zeros((batch_size, 1), dtype=np.float32))

                losses['g_fake_bin'] = \
                    self.generator_to_discriminator.train_on_batch(
                        [batched_src_imgs, batched_font_ids],
                        np.ones((batch_size, 1), dtype=np.float32))

                losses['g_const'] = \
                    self.generator_to_encoder.train_on_batch(
                        [batched_src_imgs, batched_font_ids],
                        batched_src_imgs_encoded)

                losses['g_class'] = \
                    self.generator_to_classifier.train_on_batch(
                        [batched_src_imgs, batched_font_ids],
                        batched_categorical_src_labels)

                losses['d_bin'] = losses['d_real_bin'] + losses['d_fake_bin']
                losses['d'] = losses['d_bin']
                losses['g'] = losses['g_fake_bin'] + losses['g_const'] + losses['g_class']

                self._update_losses_progress(losses, epoch_i, batch_i, batch_n)
                self._save_losses_progress_html()

                if (batch_i + 1) % save_imgs_interval == 0 or batch_i + 1 == batch_n:
                    self._save_images(batched_real_imgs, batched_fake_imgs, epoch_i, batch_i)

                if self._is_early_stopping(10):
                    print('early stop')
                    break
            else:
                continue

            self._save_model_weights(epoch_i)
            break
        self._save_losses_progress_h5()
        return losses['d'] + losses['g']

    def _labels_to_categorical(self, labels):
        return to_categorical(list(map(lambda x: ord(x) - 65, labels)), 26)

    def _init_losses_progress(self):
        self.x_time = np.array([])
        self.y_losses = dict()

    def _update_losses_progress(self, losses, epoch_i, batch_i, batch_n):
        self.x_time = np.append(self.x_time, np.array([epoch_i * batch_n + batch_i]))
        if self.y_losses == dict():
            for k, v in losses.items():
                self.y_losses[k] = np.array([v], dtype=np.float32)
        else:
            for k, v in losses.items():
                self.y_losses[k] = np.append(self.y_losses[k], np.array([v]))

    def _save_losses_progress_html(self):
        graphs = list()
        for k, v in self.y_losses.items():
            graph = go.Scatter(x=self.x_time, y=v, mode='lines', name=k)
            offline.plot([graph], filename=os.path.join(self.dst_dir_paths['losses'], '{}.html'.format(k)), auto_open=False)
            graphs.append(graph)
        offline.plot(graphs, filename=os.path.join(self.dst_dir_paths['losses'], 'all_losses.html'), auto_open=False)

    def _save_images(self, dst_imgs, generated_imgs, epoch_i, batch_i):
        concatenated_num_img = np.empty((0, 512))
        for img_i in range(dst_imgs.shape[0]):
            num_img = np.concatenate((dst_imgs[img_i], generated_imgs[img_i]), axis=1)
            num_img = num_img * 255
            num_img = np.reshape(num_img, (256, 512))
            concatenated_num_img = np.concatenate((concatenated_num_img, num_img), axis=0)
            concatenated_num_img = np.reshape(concatenated_num_img, (-1, 512))
        pil_img = Image.fromarray(np.uint8(concatenated_num_img))
        pil_img.save(os.path.join(self.dst_dir_paths['generated_imgs'], '{}_{}.png'.format(epoch_i + 1, batch_i + 1)))

    def _is_early_stopping(self, patience):
        for key in self.y_losses.keys():
            if self.y_losses[key].shape[0] > patience:
                recent_losses = self.y_losses[key][-patience:]
                if False not in (recent_losses[:] == recent_losses[0]):
                    return True
        return False

    def _save_model_weights(self, epoch_i):
        self.generator.save_weights(os.path.join(self.dst_dir_paths['model_weights'], 'gen_{}.h5'.format(epoch_i + 1)))
        self.discriminator.save_weights(os.path.join(self.dst_dir_paths['model_weights'], 'dis_{}.h5'.format(epoch_i + 1)))

    def _save_losses_progress_h5(self):
        h5file = h5py.File(os.path.join(self.dst_dir_paths['losses'], 'all_losses.h5'))
        h5file.create_dataset('x_time', data=self.x_time)
        h5file.create_dataset('y_losses', data=self.y_losses)
        h5file.flush()
        h5file.close()

    def get_generator_model(self):
        return self.generator
