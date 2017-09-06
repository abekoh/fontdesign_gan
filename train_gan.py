import os
import numpy as np
import json
from PIL import Image
import plotly.offline as py
import plotly.graph_objs as go
from scipy.signal import savgol_filter
import colorlover as cl
from tqdm import tqdm

import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical, plot_model

import models
from dataset import Dataset
from utils import concat_imgs

CAPS = [chr(i) for i in range(65, 65 + 26)]


class TrainingFontDesignGAN():

    def __init__(self, params, paths):
        self.sess = tf.Session()
        K.set_session(self.sess)

        self.params = params
        self.paths = paths
        self._set_dsts()
        self._build_models()
        self._load_dataset()
        self._prepare_training()
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

    def _build_central(self):
        if self.params.g.arch == 'dcgan':
            self.generator = models.GeneratorDCGAN_NoEmbedding(img_size=self.params.img_size,
                                                               img_dim=self.params.img_dim,
                                                               # font_embedding_n=self.params.font_embedding_n,
                                                               # char_embedding_n=self.params.char_embedding_n,
                                                               # font_embedding_rate=self.params.font_embedding_rate,
                                                               layer_n=self.params.g.layer_n,
                                                               smallest_hidden_unit_n=self.params.g.smallest_hidden_unit_n,
                                                               kernel_initializer=self.params.g.kernel_initializer,
                                                               activation=self.params.g.activation,
                                                               output_activation=self.params.g.output_activation,
                                                               is_bn=self.params.g.is_bn)
        plot_model(self.generator, to_file=os.path.join(self.paths.dst.model_visualization, 'generator.png'), show_shapes=True)
        if self.params.d.arch == 'dcgan':
            self.discriminator = models.DiscriminatorDCGAN(img_size=self.params.img_size,
                                                           img_dim=self.params.img_dim,
                                                           layer_n=self.params.d.layer_n,
                                                           smallest_hidden_unit_n=self.params.d.smallest_hidden_unit_n,
                                                           kernel_initializer=self.params.d.kernel_initializer,
                                                           activation=self.params.d.activation,
                                                           is_bn=self.params.d.is_bn)
        plot_model(self.discriminator, to_file=os.path.join(self.paths.dst.model_visualization, 'discriminator.png'), show_shapes=True)
        if hasattr(self.params, 'c'):
            self.classifier = models.Classifier(img_size=self.params.img_size,
                                                img_dim=self.params.img_dim)
            self.classifier.load_weights(self.paths.src.cls_weight_h5)
            plot_model(self.classifier, to_file=os.path.join(self.paths.dst.model_visualization, 'classifier.png'), show_shapes=True)

    def _load_dataset(self, is_shuffle=True):
        self.real_dataset = Dataset(self.paths.src.real_h5, 'r', img_size=self.params.img_size)
        self.real_dataset.set_load_data()
        self.real_dataset.set_label_ids()
        self.real_dataset.set_category_arange()
        if is_shuffle:
            self.real_dataset.shuffle()
        self.real_data_n = self.real_dataset.get_img_len()

    def _prepare_training(self):
        self.font_embedding = np.random.uniform(-1, 1, (self.params.font_embedding_n, 50))
        self.char_embedding = np.random.uniform(-1, 1, (self.params.char_embedding_n, 50))

        self.real_imgs = tf.placeholder(tf.float32, (None, self.params.img_size[0], self.params.img_size[1], self.params.img_dim), name='real_imgs')
        self.z = tf.placeholder(tf.float32, (None, 100), name='z')
        self.fake_imgs = self.generator(self.z)

        self.d_real = self.discriminator(self.real_imgs)
        self.d_fake = self.discriminator(self.fake_imgs)

        self.d_loss = - (tf.reduce_mean(self.d_real) - tf.reduce_mean(self.d_fake))
        self.g_loss = - tf.reduce_mean(self.d_fake)

        self.d_opt = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(self.d_loss, var_list=self.discriminator.trainable_weights)
        self.g_opt = tf.train.RMSPropOptimizer(learning_rate=0.00001).minimize(self.g_loss, var_list=self.generator.trainable_weights)

        if hasattr(self.params, 'c'):
            self.labels = tf.placeholder(tf.float32, (None, self.params.char_embedding_n))
            self.c_fake = self.classifier(self.fake_imgs)
            self.c_loss = - 0.1 * tf.reduce_sum(self.labels * tf.log(self.c_fake))

            self.c_opt = tf.train.RMSPropOptimizer(learning_rate=0.00001).minimize(self.c_loss, var_list=self.generator.trainable_weights)

    def _get_embedded(self, font_ids, char_ids):
        font_embedded = np.take(self.font_embedding, font_ids, axis=0)
        char_embedded = np.take(self.char_embedding, char_ids, axis=0)
        z = np.concatenate((font_embedded, char_embedded), axis=1)
        return z

    def train(self):
        self._init_metrics()

        batch_n = self.real_data_n // self.params.batch_size

        for epoch_i in tqdm(range(self.params.epoch_n)):

            for batch_i in tqdm(range(batch_n)):

                count_i = epoch_i * batch_n + batch_i

                metrics = dict()

                for i in range(self.params.critic_n):
                    d_weights = [np.clip(w, -0.01, 0.01) for w in self.discriminator.get_weights()]
                    self.discriminator.set_weights(d_weights)

                    batched_real_imgs, _ = self.real_dataset.get_random(self.params.batch_size)
                    batched_src_fonts = np.random.randint(0, self.params.font_embedding_n, (self.params.batch_size), dtype=np.int32)
                    batched_src_chars = np.random.randint(0, self.params.char_embedding_n, (self.params.batch_size), dtype=np.int32)
                    batched_z = self._get_embedded(batched_src_fonts, batched_src_chars)

                    self.sess.run(self.d_opt, feed_dict={self.z: batched_z,
                                                         self.real_imgs: batched_real_imgs,
                                                         K.learning_phase(): 1})

                batched_src_fonts = np.random.randint(0, self.params.font_embedding_n, (self.params.batch_size), dtype=np.int32)
                batched_src_chars = np.random.randint(0, self.params.char_embedding_n, (self.params.batch_size), dtype=np.int32)
                batched_z = self._get_embedded(batched_src_fonts, batched_src_chars)

                self.sess.run(self.g_opt, feed_dict={self.z: batched_z,
                                                     K.learning_phase(): 1})

                metrics['d_loss'], metrics['g_loss'] = self.sess.run([self.d_loss, self.g_loss],
                                                                     feed_dict={self.z: batched_z,
                                                                                self.real_imgs: batched_real_imgs,
                                                                                K.learning_phase(): 1})
                metrics['d_loss'] *= -1

                if hasattr(self.params, 'c'):
                    self.sess.run(self.c_opt, feed_dict={self.z: batched_z,
                                                         self.labels: to_categorical(batched_src_chars, 26),
                                                         K.learning_phase(): 1})
                    metrics['c_loss'] = self.sess.run(self.c_loss,
                                                      feed_dict={self.z: batched_z,
                                                                 self.labels: to_categorical(batched_src_chars, 26),
                                                                 K.learning_phase(): 1})

                # save metrics
                if (batch_i + 1) % self.params.save_metrics_graph_interval == 0:
                    self._update_metrics(metrics, count_i)
                    self._update_smoothed_metrics()
                    self._save_metrics_graph()

                # save images
                if (batch_i + 1) % self.params.save_imgs_interval == 0:
                    self._save_temp_imgs('{}_{}.png'.format(epoch_i + 1, batch_i + 1))

            else:
                if (epoch_i + 1) % self.params.save_weights_interval == 0:
                    self._save_model_weights(epoch_i)
                continue
            break

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

    def _init_temp_imgs_inputs(self):
        if self.params.is_all_temp:
            self.params.temp_imgs_n = self.params.font_embedding_n * self.params.char_embedding_n
            self.params.temp_col_n = self.params.char_embedding_n
            temp_batched_src_fonts = np.repeat(np.arange(0, self.params.font_embedding_n, dtype=np.int32), self.params.char_embedding_n)
            temp_batched_src_chars = np.tile(np.arange(0, self.params.char_embedding_n, dtype=np.int32), self.params.font_embedding_n)
        else:
            temp_batched_src_fonts = np.random.randint(0, self.params.font_embedding_n, (self.params.temp_imgs_n), dtype=np.int32)
            temp_batched_src_chars = np.random.randint(0, self.params.char_embedding_n, (self.params.temp_imgs_n), dtype=np.int32)
        self.temp_batched_z = self._get_embedded(temp_batched_src_fonts, temp_batched_src_chars)

    def _save_temp_imgs(self, filename):
        if not hasattr(self, 'temp_batched_z'):
            self._init_temp_imgs_inputs()
        batched_generated_imgs = self.sess.run(self.fake_imgs, feed_dict={self.z: self.temp_batched_z,
                                                                          K.learning_phase(): 1})
        row_n = self.params.temp_imgs_n // self.params.temp_col_n
        concated_img = concat_imgs(batched_generated_imgs, row_n, self.params.temp_col_n)
        concated_img = (concated_img + 1.) * 127.5
        concated_img = np.reshape(concated_img, (-1, self.params.temp_col_n * self.params.img_size[0]))
        pil_img = Image.fromarray(np.uint8(concated_img))
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
