import os
import random
import numpy as np
import h5py
from PIL import Image
import plotly.offline as offline
import plotly.graph_objs as go

import tensorflow as tf

from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import Progbar, to_categorical

import models
from dataset import Dataset
from ops import multiple_loss
from params import Params

CAPS = [chr(i) for i in range(65, 65 + 26)]


class TrainingFontDesignGAN():

    def __init__(self, params, paths):
        self.params = params
        self.paths = paths
        self._set_dsts()
        self._build_models()
        self._load_dataset()

    def _set_dsts(self):
        for path in self.paths.dst.__dict__.values():
            if not os.path.exists(path):
                os.makedirs(path)

    def _build_models(self):

        if self.params.g.arch == 'dcgan':
            self.generator = models.GeneratorDCGAN(img_size=self.params.img_size,
                                                   img_dim=self.params.img_dim,
                                                   font_embedding_n=self.params.font_embedding_n,
                                                   char_embedding_n=self.params.char_embedding_n)
        elif self.params.g.arch == 'pix2pix':
            self.generator = models.GeneratorPix2Pix(img_size=self.params.img_size,
                                                     img_dim=self.params.img_dim,
                                                     font_embedding_n=self.params.font_embedding_n)
        if self.params.d.arch == 'dcgan':
            self.discriminator = models.DiscriminatorDCGAN(img_size=self.params.img_size,
                                                           img_dim=self.params.img_dim)
        elif self.params.d.arch == 'pix2pix':
            self.discriminator = models.DiscriminatorPix2Pix(img_size=self.params.img_size,
                                                             img_dim=self.params.img_dim)
        self.discriminator_subtract = models.DiscriminatorSubtract(discriminator=self.discriminator,
                                                                   img_size=self.params.img_size,
                                                                   img_dim=self.params.img_dim,)
        self.discriminator_subtract.compile(optimizer=self.params.d.opt,
                                            loss=multiple_loss,
                                            loss_weights=self.params.d.loss_weights)

        self.discriminator.trainable = False
        self.generator_to_discriminator = Model(inputs=self.generator.input, outputs=self.discriminator(self.generator.output))
        self.generator_to_discriminator.compile(optimizer=self.params.g.opt,
                                                loss=multiple_loss,
                                                loss_weights=self.params.g.loss_weights)

        if hasattr(self.params, 'c'):
            self.classifier = models.Classifier(img_size=self.params.img_size,
                                                img_dim=self.params.img_dim, class_n=26)
            self.classifier.load_weights(self.paths.src.cls_weight_h5)
            self.classifier.trainable = False
            self.generator_to_classifier = Model(inputs=self.generator.input, outputs=self.classifier(self.generator.output))
            self.generator_to_classifier.compile(optimizer=self.params.c.opt,
                                                 loss='categorical_crossentropy',
                                                 loss_weights=self.params.c.loss_weights)

        if hasattr(self.params, 'l1'):
            self.generator.compile(optimizer=self.params.l1.opt,
                                   loss='mean_absolute_error',
                                   loss_weights=self.params.l1.loss_weights)

        if hasattr(self.params, 'e'):
            self.encoder = Model(inputs=self.generator.input[0], outputs=self.generator.get_layer('en_last').output)
            self.encoder.trainable = False
            self.generator_to_encoder = Model(inputs=self.generator.input, outputs=self.encoder(self.generator.output))
            self.generator_to_encoder.compile(optimizer=self.params.e.opt,
                                              loss='mean_squared_error',
                                              loss_weights=self.params.e.loss_weights)

    def _load_dataset(self, is_shuffle=True):
        self.real_dataset = Dataset(self.paths.src.real_h5, 'r', img_size=self.params.img_size)
        self.real_dataset.set_load_data()
        if is_shuffle:
            self.real_dataset.shuffle()
        self.real_data_n = self.real_dataset.get_img_len()
        if hasattr(self.paths.src, 'src_h5'):
            self.src_dataset = Dataset(self.paths.src.src_h5, 'r', img_size=self.params.img_size)
            self.src_dataset.set_load_data()

    def train(self):

        self._init_losses_progress()

        batch_n = self.real_data_n // self.params.batch_size

        for epoch_i in range(self.params.epoch_n):

            progbar = Progbar(batch_n)

            for batch_i in range(batch_n):

                progbar.update(batch_i + 1)

                # real imgs
                batched_real_imgs, batched_real_labels = self.real_dataset.get_batch(batch_i, self.params.batch_size)
                # src fonts info
                batched_src_fonts = np.random.randint(0, self.params.font_embedding_n, self.params.batch_size, dtype=np.int32)
                # src chars info
                if self.params.g.arch == 'dcgan':
                    batched_src_labels = [random.choice(CAPS) for i in range(self.params.batch_size)]
                    batched_src_chars = np.array([ord(i) - 65 for i in batched_src_labels], dtype=np.int32)
                elif self.params.g.arch == 'pix2pix':
                    batched_src_chars, batched_src_labels = self.src_dataset.get_selected(batched_real_labels)

                # fake imgs
                batched_fake_imgs = self.generator.predict_on_batch([batched_src_chars, batched_src_fonts])

                losses = dict()

                losses['d_wasserstein'] = 0
                for i in range(self.params.critic_n):
                    d_weights = [np.clip(w, -0.01, 0.01) for w in self.discriminator.get_weights()]
                    self.discriminator.set_weights(d_weights)

                    loss_d_wasserstein_tmp = \
                        self.discriminator_subtract.train_on_batch(
                            [batched_real_imgs, batched_fake_imgs],
                            -np.ones((self.params.batch_size, 1), dtype=np.float32))
                    losses['d_wasserstein'] += loss_d_wasserstein_tmp / self.params.critic_n

                losses['true_wasserstein'] = np.mean(self.discriminator_subtract.predict_on_batch([batched_real_imgs, batched_fake_imgs]))

                losses['g'] = 0
                losses['g_fake'] = \
                    self.generator_to_discriminator.train_on_batch(
                        [batched_src_chars, batched_src_fonts],
                        -np.ones((self.params.batch_size, 1), dtype=np.float32))
                losses['g'] += losses['g_fake']

                losses['true_g_loss'] = np.mean(self.generator_to_discriminator.predict_on_batch([batched_src_chars, batched_src_fonts]))

                if hasattr(self.params, 'c'):
                    batched_categorical_src_labels = self._labels_to_categorical(batched_src_labels)
                    losses['g_class'] = \
                        self.generator_to_classifier.train_on_batch(
                            [batched_src_chars, batched_src_fonts],
                            batched_categorical_src_labels)
                    losses['g'] += losses['g_class']

                if hasattr(self.params, 'e'):
                    batched_src_chars_encoded = self.encoder.predict_on_batch(batched_src_chars)
                    losses['g_const'] = \
                        self.generator_to_encoder.train_on_batch(
                            [batched_src_chars, batched_src_fonts],
                            batched_src_chars_encoded)
                    losses['g'] += losses['g_const']

                if hasattr(self.params, 'l1'):
                    losses['g_l1'] = \
                        self.generator.train_on_batch(
                            [batched_src_chars, batched_src_fonts],
                            batched_real_imgs)
                    losses['g'] += losses['g_l1']

                if (batch_i + 1) % self.params.save_graphs_interval == 0:
                    self._update_losses_progress(losses, epoch_i * batch_n + batch_i)
                    self._save_losses_progress_html()

                if (batch_i + 1) % self.params.save_imgs_interval == 0:
                    self._save_images(batched_real_imgs, batched_fake_imgs, epoch_i, batch_i)

                if self._is_early_stopping(self.params.early_stopping_n):
                    print('early stop')
                    break
            else:
                continue

            if (epoch_i + 1) % self.params.save_weights_interval == 0:
                self._save_model_weights(epoch_i)
            break
        # self._save_losses_progress_h5()

    def _labels_to_categorical(self, labels):
        return to_categorical(list(map(lambda x: ord(x) - 65, labels)), 26)

    def _init_losses_progress(self):
        self.x_time = np.array([])
        self.y_losses = dict()

    def _update_losses_progress(self, losses, x):
        self.x_time = np.append(self.x_time, np.array([x]))
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
            offline.plot([graph], filename=os.path.join(self.paths.dst.losses, '{}.html'.format(k)), auto_open=False)
            graphs.append(graph)
        offline.plot(graphs, filename=os.path.join(self.paths.dst.losses, 'all_losses.html'), auto_open=False)

    def _save_images(self, dst_imgs, generated_imgs, epoch_i, batch_i):
        concatenated_num_img = np.empty((0, self.params.img_size[1] * 2))
        for img_i in range(dst_imgs.shape[0]):
            num_img = np.concatenate((dst_imgs[img_i], generated_imgs[img_i]), axis=1)
            num_img = num_img * 255
            num_img = np.reshape(num_img, (self.params.img_size[0], self.params.img_size[1] * 2))
            concatenated_num_img = np.concatenate((concatenated_num_img, num_img), axis=0)
            concatenated_num_img = np.reshape(concatenated_num_img, (-1, self.params.img_size[1] * 2))
        pil_img = Image.fromarray(np.uint8(concatenated_num_img))
        pil_img.save(os.path.join(self.paths.dst.generated_imgs, '{}_{}.png'.format(epoch_i + 1, batch_i + 1)))

    def _is_early_stopping(self, patience):
        for key in self.y_losses.keys():
            if self.y_losses[key].shape[0] > patience:
                recent_losses = self.y_losses[key][-patience:]
                if False not in (recent_losses[:] == recent_losses[0]):
                    return True
        return False

    def _save_model_weights(self, epoch_i):
        self.generator.save_weights(os.path.join(self.paths.dst.model_weights, 'gen_{}.h5'.format(epoch_i + 1)))
        self.discriminator.save_weights(os.path.join(self.paths.dst.model_weights, 'dis_{}.h5'.format(epoch_i + 1)))

    def _save_losses_progress_h5(self):
        h5file = h5py.File(os.path.join(self.paths.dst.losses, 'all_losses.h5'))
        h5file.create_dataset('x_time', data=self.x_time)
        h5file.create_dataset('y_losses', data=self.y_losses)
        h5file.flush()
        h5file.close()


if __name__ == '__main__':
    params = Params(d={
        'img_size': (256, 256),
        'img_dim': 1,
        'font_embedding_n': 40,
        'char_embedding_n': 26,
        'epoch_n': 1000,
        'batch_size': 32,
        'critic_n': 5,
        'early_stopping_n': 10,
        'save_graphs_interval': 1,
        'save_imgs_interval': 10,
        'save_weights_interval': 5,
        'g': Params({
            'arch': 'pix2pix',
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [1.],
        }),
        'd': Params({
            'arch': 'pix2pix',
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [1.],
        }),
        # 'c': Params({
        #     'opt': RMSprop(lr=0.00005),
        #     'loss_weights': [1.]
        # }),
        'l1': Params({
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [100.]
        }),
        'e': Params({
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [15.]
        })
    })

    dst_root = 'output/0727'

    paths = Params({
        'src': Params({
            'real_h5': 'src/fonts_200_caps_256x256.h5',
            'src_h5': 'src/arial.h5',
            # 'cls_weight_h5': 'output_classifier/classifier_weights_20(train=0.936397172634403,test=0.9258828996282528).h5'
        }),
        'dst': Params({
            'generated_imgs': '{}/generated_imgs'.format(dst_root),
            'model_weights': '{}/model_weights'.format(dst_root),
            'losses': '{}/losses'.format(dst_root)
        })
    })
    gan = TrainingFontDesignGAN(params, paths)
    gan.train()
