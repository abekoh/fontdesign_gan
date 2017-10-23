import os

import tensorflow as tf
from keras.utils import to_categorical
from tqdm import tqdm

from models import Classifier
from dataset import Dataset

FLAGS = tf.app.flags.FLAGS


class TrainingClassifier():

    def __init__(self):
        global FLAGS

    def setup(self):
        self._make_dirs()
        self._prepare_training()
        self._load_dataset()

    def _make_dirs(self):
        os.mkdir(FLAGS.dst_classifier_root)
        os.mkdir(FLAGS.dst_classifier_log)

    def _prepare_training(self):
        self.classifier = Classifier(img_size=(FLAGS.img_width, FLAGS.img_height),
                                     img_dim=FLAGS.img_dim,
                                     k_size=FLAGS.c_k_size,
                                     class_n=26,
                                     smallest_unit_n=FLAGS.c_smallest_unit_n)
        self.imgs = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim), name='imgs')
        self.labels = tf.placeholder(tf.float32, (FLAGS.batch_size, 26), name='labels')
        classified = self.classifier(self.imgs)
        self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=classified))
        self.c_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.c_loss)
        correct_pred = tf.equal(tf.argmax(classified, 1), tf.argmax(self.labels, 1))
        self.c_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids)
        )
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def _load_dataset(self):
        self.dataset = Dataset(FLAGS.src_real_h5, 'r', (FLAGS.img_width, FLAGS.img_height), img_dim=FLAGS.img_dim)
        self.dataset.set_load_data(train_rate=FLAGS.train_rate)
        if FLAGS.is_shuffle:
            self.dataset.shuffle()
        self.train_data_n = self.dataset.get_img_len()
        self.test_data_n = self.dataset.get_img_len(is_test=True)

    def train(self):
        train_batch_n = self.train_data_n // FLAGS.batch_size
        test_batch_n = self.test_data_n // FLAGS.batch_size
        for epoch_i in tqdm(range(FLAGS.epoch_n)):
            # train
            losses, accs = list(), list()
            for batch_i in tqdm(range(train_batch_n)):
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, FLAGS.batch_size)
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
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, FLAGS.batch_size, is_test=True)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                loss, acc = self.sess.run([self.c_loss, self.c_acc],
                                          feed_dict={self.imgs: batched_imgs,
                                                     self.labels: batched_categorical_labels})
                losses.append(loss)
                accs.append(acc)
            test_loss_avg = sum(losses) / len(losses)
            test_acc_avg = sum(accs) / len(accs)
            print('[test] loss: {}, acc: {}\n'.format(test_loss_avg, test_acc_avg))
            self.saver.save(self.sess, os.path.join(FLAGS.dst_classifier_log, 'result_{}.ckpt'.format(epoch_i)))

    def _labels_to_categorical(self, labels):
        return to_categorical(list(map(lambda x: ord(x) - 65, labels)), 26)
