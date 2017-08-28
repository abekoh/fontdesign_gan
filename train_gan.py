import os
import random
import numpy as np
import h5py
import json
from PIL import Image
import plotly.offline as py
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import colorlover as cl

import tensorflow as tf
from keras.models import Model
from keras.utils import Progbar, to_categorical, plot_model

import models
from dataset import Dataset
from ops import hamming_error_inv, multiple_loss

CAPS = [chr(i) for i in range(65, 65 + 26)]


class TrainingFontDesignGAN():

    def __init__(self, params, paths):
        self.params = params
        self.paths = paths
        self._set_dsts()
        self._build_models()
        self._load_dataset()
        self._save_params()

    def _set_dsts(self):
        for path in self.paths.dst.__dict__.values():
            if not os.path.exists(path):
                os.makedirs(path)

    def _save_params(self):
        with open(os.path.join(self.paths.dst.root, 'params.txt'), 'w') as f:
            json.dump(self.params.to_dict(), f, indent=4)
        with open(os.path.join(self.paths.dst.root, 'paths.txt'), 'w') as f:
            json.dump(self.paths.to_dict(), f, indent=4)

    def _build_models(self):

        self._build_central()

        if hasattr(self.params, 'dc') and hasattr(self.params, 'gc'):
            self._build_fontclassifier()

        if hasattr(self.params, 'c'):
            self._build_classifier()

        if hasattr(self.params, 'l1'):
            self.generator.compile(optimizer=self.params.l1.opt,
                                   loss='mean_absolute_error',
                                   loss_weights=self.params.l1.loss_weights)

        if hasattr(self.params, 'v'):
            self.generator.compile(optimizer=self.params.v.opt,
                                   loss=hamming_error_inv,
                                   loss_weights=self.params.v.loss_weights)

        if hasattr(self.params, 'e'):
            self._build_encoder()

    def _build_central(self):

        if self.params.g.arch == 'dcgan':
            self.generator = models.GeneratorDCGAN(img_size=self.params.img_size,
                                                   img_dim=self.params.img_dim,
                                                   font_embedding_n=self.params.font_embedding_n,
                                                   char_embedding_n=self.params.char_embedding_n,
                                                   font_embedding_rate=self.params.font_embedding_rate,
                                                   layer_n=self.params.g.layer_n,
                                                   smallest_hidden_unit_n=self.params.g.smallest_hidden_unit_n,
                                                   kernel_initializer=self.params.g.kernel_initializer,
                                                   activation=self.params.g.activation,
                                                   is_bn=self.params.g.is_bn)
        elif self.params.g.arch == 'pix2pix':
            self.generator = models.GeneratorPix2Pix(img_size=self.params.img_size,
                                                     img_dim=self.params.img_dim,
                                                     font_embedding_n=self.params.font_embedding_n)
        plot_model(self.generator, to_file=os.path.join(self.paths.dst.model_visualization, 'generator.png'), show_shapes=True)
        if self.params.d.arch == 'dcgan':
            self.discriminator = models.DiscriminatorDCGAN(img_size=self.params.img_size,
                                                           img_dim=self.params.img_dim,
                                                           layer_n=self.params.d.layer_n,
                                                           smallest_hidden_unit_n=self.params.d.smallest_hidden_unit_n,
                                                           kernel_initializer=self.params.d.kernel_initializer,
                                                           activation=self.params.d.activation,
                                                           is_bn=self.params.d.is_bn)
        elif self.params.d.arch == 'pix2pix':
            self.discriminator = models.DiscriminatorPix2Pix(img_size=self.params.img_size,
                                                             img_dim=self.params.img_dim,
                                                             font_embedding_n=self.params.font_embedding_n)
        plot_model(self.discriminator, to_file=os.path.join(self.paths.dst.model_visualization, 'discriminator.png'), show_shapes=True)
        self.discriminator_bin_sub = models.DiscriminatorBinarizeSubtract(discriminator=self.discriminator,
                                                                          img_size=self.params.img_size,
                                                                          img_dim=self.params.img_dim)
        self.discriminator_bin_sub.compile(optimizer=self.params.d.opt,
                                           loss=multiple_loss,
                                           loss_weights=self.params.d.loss_weights)
        self.discriminator_bin = models.DiscriminatorBinarize(discriminator=self.discriminator,
                                                              img_size=self.params.img_size,
                                                              img_dim=self.params.img_dim)
        self.discriminator_bin.trainable = False
        self.generator_to_discriminator_bin = Model(inputs=self.generator.input, outputs=self.discriminator_bin(self.generator.output))
        self.generator_to_discriminator_bin.compile(optimizer=self.params.g.opt,
                                                    loss=multiple_loss,
                                                    loss_weights=self.params.g.loss_weights)
        self.discriminator_bin.trainable = True

    def _build_fontclassifier(self):
        self.discriminator_cat = models.DiscriminatorCategorize(discriminator=self.discriminator,
                                                                img_size=self.params.img_size,
                                                                img_dim=self.params.img_dim,
                                                                font_embedding_n=self.params.font_embedding_n)
        self.discriminator_cat.compile(optimizer=self.params.dc.opt,
                                       loss='categorical_crossentropy',
                                       loss_weights=self.params.dc.loss_weights)

        self.discriminator_cat.trainable = False
        self.generator_to_discriminator_cat = Model(inputs=self.generator.input, outputs=self.discriminator_cat(self.generator.output))
        self.generator_to_discriminator_cat.compile(optimizer=self.params.gc.opt,
                                                    loss='categorical_crossentropy',
                                                    loss_weights=self.params.gc.loss_weights)

    def _build_classifier(self):
        self.classifier = models.Classifier(img_size=self.params.img_size,
                                            img_dim=self.params.img_dim, class_n=26)
        plot_model(self.classifier, to_file=os.path.join(self.paths.dst.model_visualization, 'classifier.png'), show_shapes=True)
        self.classifier.load_weights(self.paths.src.cls_weight_h5)
        self.classifier.trainable = False
        self.generator_to_classifier = Model(inputs=self.generator.input, outputs=self.classifier(self.generator.output))
        self.generator_to_classifier.compile(optimizer=self.params.c.opt,
                                             loss='categorical_crossentropy',
                                             loss_weights=self.params.c.loss_weights)

    def _build_encoder(self):
        self.encoder = Model(inputs=self.generator.input[0], outputs=self.generator.get_layer('en_last').output)
        self.generator_to_encoder = Model(inputs=self.generator.input, outputs=self.encoder(self.generator.output))
        self.encoder.trainable = False
        self.generator_to_encoder.compile(optimizer=self.params.e.opt,
                                          loss='mean_squared_error',
                                          loss_weights=self.params.e.loss_weights)

    def _load_dataset(self, is_shuffle=True):
        self.real_dataset = Dataset(self.paths.src.real_h5, 'r', img_size=self.params.img_size)
        self.real_dataset.set_load_data()
        self.real_dataset.set_category_arange()
        if is_shuffle:
            self.real_dataset.shuffle()
        self.real_data_n = self.real_dataset.get_img_len()
        if hasattr(self.paths.src, 'src_h5'):
            self.src_dataset = Dataset(self.paths.src.src_h5, 'r', img_size=self.params.img_size)
            self.src_dataset.set_load_data()

    def train(self):

        self._init_metrics()

        batch_n = self.real_data_n // self.params.batch_size

        # self.tb_writer = tf.summary.FileWriter(self.paths.dst.tensorboard_log)

        for epoch_i in range(self.params.epoch_n):

            progbar = Progbar(batch_n)

            for batch_i in range(batch_n):

                progbar.update(batch_i + 1)
                count_i = epoch_i * batch_n + batch_i

                # real imgs
                batched_real_imgs, batched_real_labels, batched_real_cats = self.real_dataset.get_batch(batch_i, self.params.batch_size, is_cat=True)
                # src fonts info
                if hasattr(self.params, 'dc'):
                    batched_src_fonts = np.array(batched_real_cats)
                else:
                    batched_src_fonts = np.random.randint(0, self.params.font_embedding_n, self.params.batch_size, dtype=np.int32)

                # src chars info
                if self.params.g.arch == 'dcgan':
                    batched_src_labels = [random.choice(CAPS) for i in range(self.params.batch_size)]
                    batched_src_chars = np.array([ord(i) - 65 for i in batched_src_labels], dtype=np.int32)
                elif self.params.g.arch == 'pix2pix':
                    batched_src_chars, batched_src_labels = self.src_dataset.get_selected(batched_real_labels)

                # fake imgs
                batched_fake_imgs = self.generator.predict_on_batch([batched_src_chars, batched_src_fonts])

                metrics = dict()

                metrics['d_wasserstein'] = 0.
                for i in range(self.params.critic_n):
                    d_weights = [np.clip(w, -0.01, 0.01) for w in self.discriminator.get_weights()]
                    self.discriminator.set_weights(d_weights)

                    d_wasserstein_tmp = \
                        self.discriminator_bin_sub.train_on_batch(
                            [batched_real_imgs, batched_fake_imgs],
                            -np.ones((self.params.batch_size, 1), dtype=np.float32))
                    metrics['d_wasserstein'] += d_wasserstein_tmp / self.params.critic_n
                metrics['d_wasserstein'] *= -1

                metrics['g_fake_bin'] = \
                    self.generator_to_discriminator_bin.train_on_batch(
                        [batched_src_chars, batched_src_fonts],
                        np.ones((self.params.batch_size, 1), dtype=np.float32))

                if hasattr(self.params, 'dc'):
                    metrics['d_real_cat'] = \
                        self.discriminator_cat.train_on_batch(
                            batched_real_imgs,
                            to_categorical(batched_src_fonts, self.params.font_embedding_n))
                    metrics['d_fake_cat'] = \
                        self.discriminator_cat.train_on_batch(
                            batched_fake_imgs,
                            to_categorical(batched_src_fonts, self.params.font_embedding_n))

                if hasattr(self.params, 'gc'):
                    metrics['g_fake_cat'] = \
                        self.generator_to_discriminator_cat.train_on_batch(
                            [batched_src_chars, batched_src_fonts],
                            to_categorical(batched_src_fonts, self.params.font_embedding_n))

                if hasattr(self.params, 'c'):
                    batched_categorical_src_labels = self._labels_to_categorical(batched_src_labels)
                    metrics['g_class'] = \
                        self.generator_to_classifier.train_on_batch(
                            [batched_src_chars, batched_src_fonts],
                            batched_categorical_src_labels)

                if hasattr(self.params, 'l1'):
                    metrics['g_l1'] = \
                        self.generator.train_on_batch(
                            [batched_src_chars, batched_src_fonts],
                            batched_real_imgs)

                if hasattr(self.params, 'v'):
                    src_char, _ = self.src_dataset.get_selected([random.choice(CAPS)])
                    v_src_chars = np.concatenate([src_char] * self.params.font_embedding_n).reshape(-1, 256, 256, 1)
                    v_src_fonts = np.arange(0, self.params.font_embedding_n, dtype=np.int32)
                    v_fake_fonts = self.generator.predict_on_batch([v_src_chars, v_src_fonts])
                    v_mean_fonts = np.array([np.mean(v_fake_fonts, axis=0)] * self.params.font_embedding_n)
                    metrics['v'] = \
                        self.generator.train_on_batch(
                            [v_src_chars, v_src_fonts],
                            v_mean_fonts)
                    metrics['v'] *= -1

                if hasattr(self.params, 'e'):
                    batched_src_chars_encoded = self.encoder.predict_on_batch(batched_src_chars)
                    metrics['g_const'] = \
                        self.generator_to_encoder.train_on_batch(
                            [batched_src_chars, batched_src_fonts],
                            batched_src_chars_encoded)

                # save metrics
                # self._update_tensorboard_metrics(metrics, count_i)
                if (batch_i + 1) % self.params.save_metrics_graph_interval == 0:
                    self._update_metrics(metrics, count_i)
                    self._update_smoothed_metrics()
                    self._save_metrics_graph()

                # save images
                if (batch_i + 1) % self.params.save_imgs_interval == 0:
                    self._save_images(batched_real_imgs, batched_fake_imgs, '{}_{}.png'.format(epoch_i + 1, batch_i + 1))
                    if hasattr(self.params, 'v'):
                        self._save_images(v_fake_fonts, v_mean_fonts, 'mean_{}_{}.png'.format(epoch_i + 1, batch_i + 1))

                if hasattr(self.params, 'early_stopping_n') and self._is_early_stopping(self.params.early_stopping_n):
                    print('early stop')
                    break

            else:
                if (epoch_i + 1) % self.params.save_weights_interval == 0:
                    self._save_model_weights(epoch_i)
                continue
            break
        # self._save_metrics_progress_h5()

    def _make_another_random_array(self, from_n, to_n, src_array):
        dst_array = np.array([], dtype=np.int32)
        for num in src_array:
            rand_num = num
            count = 0
            while rand_num == num:
                rand_num = np.random.randint(from_n, to_n, dtype=np.int32)
                count += 1
            dst_array = np.append(dst_array, rand_num)
        return dst_array

    def _labels_to_categorical(self, labels):
        return to_categorical(list(map(lambda x: ord(x) - 65, labels)), 26)

    def _update_tensorboard_metrics(self, metrics, count_i):
        for name, value in metrics.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.tb_writer.add_summary(summary, count_i)
            self.tb_writer.flush()

    def _init_metrics(self):
        self.metrics = dict()
        self.smoothed_metrics = dict()

    def _update_metrics(self, metrics, count_i):
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = np.array([[count_i], [metrics[k]]])
            else:
                self.metrics[k] = np.concatenate((self.metrics[k], np.array([[count_i], [metrics[k]]])), axis=1)

    def _update_smoothed_metrics(self):
        for k, v in self.metrics.items():
            window_length = len(v[0]) // 4
            if window_length % 2 == 0:
                window_length += 1
            if window_length <= 3:
                continue
            filtered = savgol_filter(v[1], window_length, 3)
            self.smoothed_metrics[k] = np.array([v[0], filtered])

    def _save_metrics_graph(self):
        all_graphs = list()
        metrics_n = len(self.metrics) + 1
        for i, k in enumerate(self.metrics.keys()):
            graph = go.Scatter(x=self.metrics[k][0], y=self.metrics[k][1], mode='lines', name=k, line=dict(dash='dot', color=cl.scales[str(metrics_n)]['qual']['Paired'][i]))
            graphs = [graph]
            if k in self.smoothed_metrics:
                smoothed_graph = go.Scatter(x=self.smoothed_metrics[k][0], y=self.smoothed_metrics[k][1], mode='lines', name=k + '_smoothed', line=dict(color=cl.scales[str(metrics_n)]['qual']['Paired'][i]))
                graphs.append(smoothed_graph)
            py.plot(graphs, filename=os.path.join(self.paths.dst.metrics, '{}.html'.format(k)), auto_open=False)
            all_graphs.extend(graphs)
        py.plot(all_graphs, filename=os.path.join(self.paths.dst.metrics, 'all_metrics.html'), auto_open=self.params.is_auto_open)
        self.params.is_auto_open = False

    def _save_images(self, dst_imgs, generated_imgs, filename):
        concatenated_num_img = np.empty((0, self.params.img_size[1] * 2))
        for img_i in range(dst_imgs.shape[0]):
            num_img = np.concatenate((dst_imgs[img_i], generated_imgs[img_i]), axis=1)
            num_img = (num_img + 1.) * 127.5
            num_img = np.reshape(num_img, (self.params.img_size[0], self.params.img_size[1] * 2))
            concatenated_num_img = np.concatenate((concatenated_num_img, num_img), axis=0)
            concatenated_num_img = np.reshape(concatenated_num_img, (-1, self.params.img_size[1] * 2))
        pil_img = Image.fromarray(np.uint8(concatenated_num_img))
        pil_img.save(os.path.join(self.paths.dst.generated_imgs, filename))

    def _is_early_stopping(self, patience):
        for key in self.y_metrics.keys():
            if self.y_metrics[key].shape[0] > patience:
                recent_metrics = self.y_metrics[key][-patience:]
                if False not in (recent_metrics[:] == recent_metrics[0]):
                    return True
        return False

    def _save_model_weights(self, epoch_i):
        self.generator.save_weights(os.path.join(self.paths.dst.model_weights, 'gen_{}.h5'.format(epoch_i + 1)))
        self.discriminator.save_weights(os.path.join(self.paths.dst.model_weights, 'dis_{}.h5'.format(epoch_i + 1)))

    def _save_metrics_progress_h5(self):
        h5file = h5py.File(os.path.join(self.paths.dst.metrics, 'all_metrics.h5'))
        h5file.create_dataset('x_time', data=self.x_time)
        h5file.create_dataset('y_metrics', data=self.y_metrics)
        h5file.flush()
        h5file.close()

    def get_last_metric(self, key):
        return self.metrics[key][1][-1]

    def get_metric_decrease(self, key, n):
        if key not in self.smoothed_metrics or len(self.smoothed_metrics[key][1]) < n:
            return 0
        decrease_sum = 0
        for i in range(n):
            decrease_sum += self.smoothed_metrics[key][1][- (n - i)] - self.smoothed_metrics[key][1][- (n - i + 1)]
        decrease_avg = decrease_sum / n
        return decrease_avg
