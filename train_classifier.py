import os
import json

import tensorflow as tf
from keras.utils import to_categorical
from tqdm import tqdm

from models import Classifier
from dataset import Dataset


class TrainingClassifier():

    def __init__(self, params, paths):
        self.params = params
        self.paths = paths
        self._set_outputs()
        self._save_params()
        self._prepare_training()
        self._load_dataset()

    def _set_outputs(self):
        if not os.path.exists(self.paths.dst.root):
            os.makedirs(self.paths.dst.root)

    def _save_params(self):
        with open(os.path.join(self.paths.dst.root, 'params.txt'), 'w') as f:
            json.dump(self.params.to_dict(), f, indent=4)
        with open(os.path.join(self.paths.dst.root, 'paths.txt'), 'w') as f:
            json.dump(self.paths.to_dict(), f, indent=4)

    def _prepare_training(self):
        self.classifier = Classifier(img_size=self.params.img_size,
                                     img_dim=self.params.img_dim,
                                     k_size=self.params.k_size,
                                     class_n=self.params.class_n,
                                     smallest_unit_n=self.params.smallest_unit_n)
        self.imgs = tf.placeholder(tf.float32, (self.params.batch_size, self.params.img_size[0], self.params.img_size[1], self.params.img_dim), name='imgs')
        self.labels = tf.placeholder(tf.float32, (self.params.batch_size, self.params.class_n), name='labels')
        classified = self.classifier(self.imgs)
        self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=classified))
        self.c_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.c_loss)
        correct_pred = tf.equal(tf.argmax(classified, 1), tf.argmax(self.labels, 1))
        self.c_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def _load_dataset(self):
        self.dataset = Dataset(self.paths.src.fonts, 'r', self.params.img_size, img_dim=self.params.img_dim)
        self.dataset.set_load_data(train_rate=self.params.train_rate)
        if self.params.is_shuffle:
            self.dataset.shuffle()
        self.train_data_n = self.dataset.get_img_len()
        self.test_data_n = self.dataset.get_img_len(is_test=True)

    def train(self):

        train_batch_n = self.train_data_n // self.params.batch_size
        test_batch_n = self.test_data_n // self.params.batch_size
        for epoch_i in tqdm(range(self.params.epoch_n)):
            # train
            losses, accs = list(), list()
            for batch_i in tqdm(range(train_batch_n)):
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, self.params.batch_size)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                _, loss, acc = self.sess.run([self.c_opt, self.c_loss, self.c_acc],
                                             feed_dict={self.imgs: batched_imgs,
                                                        self.labels: batched_categorical_labels})
                losses.append(loss)
                accs.append(acc)
            train_loss_avg = sum(losses) / len(losses)
            train_acc_avg = sum(accs) / len(accs)
            print('[train] loss: {}, acc: {}\n'.format(train_loss_avg, train_acc_avg))
            # test
            accs = list()
            for batch_i in tqdm(range(test_batch_n)):
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, self.params.batch_size, is_test=True)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                loss, acc = self.sess.run([self.c_loss, self.c_acc],
                                          feed_dict={self.imgs: batched_imgs,
                                                     self.labels: batched_categorical_labels})
                losses.append(loss)
                accs.append(acc)
            test_loss_avg = sum(losses) / len(losses)
            test_acc_avg = sum(accs) / len(accs)
            print('[test] loss: {}, acc: {}\n'.format(test_loss_avg, test_acc_avg))
            self.saver.save(self.sess, os.path.join(self.paths.dst.log, 'result_{}.ckpt'.format(epoch_i)))

    def _labels_to_categorical(self, labels):
        return to_categorical(list(map(lambda x: ord(x) - 65, labels)), 26)
