import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from dataset import Dataset
from models import Classifier
from ops import average_gradients

FLAGS = tf.app.flags.FLAGS


class TrainingClassifier():

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._prepare_training()
        self._load_dataset()

    def _setup_dirs(self):
        '''
        Setup output directories
        '''
        if not os.path.exists(FLAGS.dst_classifier):
            os.makedirs(FLAGS.dst_classifier)
        self.dst_log = os.path.join(FLAGS.dst_classifier, 'log')
        if not os.path.exists(self.dst_log):
            os.mkdir(self.dst_log)

    def _prepare_training(self):
        '''
        Prepare Training
        '''
        self.gpu_n = len(FLAGS.gpu_ids.split(','))
        with tf.device('/cpu:0'):
            self.imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='imgs')
            self.labels = tf.placeholder(tf.float32, (FLAGS.batch_size, 26), name='labels')
            self.is_train = tf.placeholder(tf.bool, name='is_train')
            c_opt = tf.train.AdamOptimizer(learning_rate=0.001)

        c_loss = [0] * self.gpu_n
        c_acc = [0] * self.gpu_n
        c_grads = [0] * self.gpu_n
        is_not_first = False

        for i in range(self.gpu_n):
            with tf.device('/gpu:{}'.format(i)):
                classifier = Classifier(img_size=(FLAGS.img_width, FLAGS.img_height),
                                        img_dim=FLAGS.img_dim,
                                        k_size=3,
                                        class_n=26,
                                        smallest_unit_n=64)
                classified = classifier(self.imgs, is_train=self.is_train, is_reuse=is_not_first)
                c_loss[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=classified))
                correct_pred = tf.equal(tf.argmax(classified, 1), tf.argmax(self.labels, 1))
                c_acc[i] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                c_vars = [var for var in tf.trainable_variables() if 'classifier' in var.name]
                c_grads[i] = c_opt.compute_gradients(c_loss[i], var_list=c_vars)
            is_not_first = True

        with tf.device('/cpu:0'):
            self.c_loss = sum(c_loss) / len(c_loss)
            self.c_acc = sum(c_acc) / len(c_acc)
            avg_c_grads = average_gradients(c_grads)
            self.c_train = c_opt.apply_gradients(avg_c_grads)

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def _load_dataset(self):
        self.dataset = Dataset(FLAGS.font_h5, 'r', (FLAGS.img_width, FLAGS.img_height), img_dim=FLAGS.img_dim, is_mem=True)
        self.dataset.set_load_data(train_rate=FLAGS.train_rate)
        self.dataset.shuffle()
        self.train_data_n = self.dataset.get_img_len()
        self.test_data_n = self.dataset.get_img_len(is_test=True)

    def train(self):
        train_batch_n = self.train_data_n // FLAGS.batch_size
        test_batch_n = self.test_data_n // FLAGS.batch_size
        for epoch_i in tqdm(range(FLAGS.c_epoch_n)):
            # train
            losses, accs = list(), list()
            for batch_i in tqdm(range(train_batch_n)):
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, FLAGS.batch_size)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                _, loss, acc = self.sess.run([self.c_train, self.c_loss, self.c_acc],
                                             feed_dict={self.imgs: batched_imgs,
                                                        self.labels: batched_categorical_labels,
                                                        self.is_train: True})
                losses.append(loss)
                accs.append(acc)
            train_loss_avg = sum(losses) / len(losses)
            train_acc_avg = sum(accs) / len(accs)
            print('[train] loss: {}, acc: {}\n'.format(train_loss_avg, train_acc_avg))
            # test
            accs = list()
            for batch_i in tqdm(range(test_batch_n)):
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, FLAGS.batch_size, is_test=True)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                loss, acc = self.sess.run([self.c_loss, self.c_acc],
                                          feed_dict={self.imgs: batched_imgs,
                                                     self.labels: batched_categorical_labels,
                                                     self.is_train: False})
                losses.append(loss)
                accs.append(acc)
            test_loss_avg = sum(losses) / len(losses)
            test_acc_avg = sum(accs) / len(accs)
            print('[test] loss: {}, acc: {}\n'.format(test_loss_avg, test_acc_avg))
            self.saver.save(self.sess, os.path.join(self.dst_log, 'result_{}.ckpt'.format(epoch_i)))

    def _labels_to_categorical(self, labels):
        return np.eye(26)[list(map(lambda x: ord(x) - 65, labels))]
