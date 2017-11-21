import os
import csv
import json

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
        self._save_flags()
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

    def _save_flags(self):
        '''
        Save FLAGS as JSON
        '''
        with open(os.path.join(self.dst_log, 'flags.json'), 'w') as f:
            json.dump(FLAGS.__dict__['__flags'], f, indent=4)

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
        label_corrects = [0] * self.gpu_n
        label_ns = [0] * self.gpu_n
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
                label_indecies = tf.argmax(self.labels, 1)
                correct_pred = tf.equal(tf.argmax(classified, 1), label_indecies)
                c_acc[i] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                if FLAGS.labelacc:
                    masked_label_indecies = tf.boolean_mask(label_indecies, correct_pred)
                    label_corrects[i] = tf.reduce_sum(tf.one_hot(masked_label_indecies, 26), axis=0)
                    label_ns[i] = tf.reduce_sum(tf.one_hot(label_indecies, 26), axis=0)

                c_vars = [var for var in tf.trainable_variables() if 'classifier' in var.name]
                c_grads[i] = c_opt.compute_gradients(c_loss[i], var_list=c_vars)
            is_not_first = True

        with tf.device('/cpu:0'):
            self.c_loss = sum(c_loss) / len(c_loss)
            self.c_acc = sum(c_acc) / len(c_acc)
            if FLAGS.labelacc:
                self.c_acc_by_labels = sum(label_corrects) / sum(label_ns)
            avg_c_grads = average_gradients(c_grads)
            self.c_train = c_opt.apply_gradients(avg_c_grads)

        sess_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu_ids),
            allow_soft_placement=FLAGS.labelacc,
            log_device_placement=FLAGS.labelacc
        )
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.dst_log)
        if checkpoint:
            saver_resume = tf.train.Saver()
            saver_resume.restore(self.sess, checkpoint.model_checkpoint_path)
            self.epoch_start = int(checkpoint.model_checkpoint_path.split('-')[-1])
            print('restore ckpt')
        else:
            self.sess.run(tf.global_variables_initializer())
            self.epoch_start = 0

    def _load_dataset(self):
        self.dataset = Dataset(FLAGS.font_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.dataset.set_load_data(train_rate=FLAGS.train_rate)
        self.dataset.shuffle()
        self.train_data_n = self.dataset.get_data_n()
        self.test_data_n = self.dataset.get_data_n(is_test=True)

    def _run(self, is_test=False):
        loss_avg = 0
        acc_avg = 0
        acc_by_labels = np.zeros((26))
        if is_test:
            batch_n = self.test_data_n // FLAGS.batch_size
            fetches = [self.c_loss, self.c_acc]
        else:
            batch_n = self.train_data_n // FLAGS.batch_size
            fetches = [self.c_train, self.c_loss, self.c_acc]
        if FLAGS.labelacc:
            fetches.append(self.c_acc_by_labels)
        for batch_i in tqdm(range(batch_n)):
            imgs, labels = self.dataset.get_batch(batch_i, FLAGS.batch_size, is_label=True, is_test=is_test)
            categorical_labels = np.eye(26)[self.dataset.get_ids_from_labels(labels)]
            rets = self.sess.run(fetches,
                                 feed_dict={self.imgs: imgs,
                                            self.labels: categorical_labels,
                                            self.is_train: not is_test})
            if not is_test:
                del rets[0]
            loss_avg += rets[0] / batch_n
            acc_avg += rets[1] / batch_n
            if FLAGS.labelacc:
                acc_by_labels = acc_by_labels + np.nan_to_num(rets[2]) / batch_n
        results = [loss_avg, acc_avg]
        if FLAGS.labelacc:
            return results + acc_by_labels.tolist()
        return results

    def train(self):
        train_metrics = list()
        test_metrics = list()
        try:
            for epoch_i in tqdm(range(self.epoch_start, FLAGS.c_epoch_n), initial=self.epoch_start, total=FLAGS.c_epoch_n):
                train_rets = self._run(is_test=False)
                print('[train] loss: {}, accuracy: {}'.format(train_rets[0], train_rets[1]))
                test_rets = self._run(is_test=True)
                print('[test] loss: {}, accuracy: {}'.format(test_rets[0], test_rets[1]))

                self.saver.save(self.sess, os.path.join(self.dst_log, 'result.ckpt'), global_step=epoch_i + 1)
                train_metrics.append([epoch_i + 1] + train_rets)
                test_metrics.append([epoch_i + 1] + test_rets)
        except KeyboardInterrupt:
            print('cancelled. but write csv...')
        finally:
            self._write_csv('train', train_metrics)
            self._write_csv('test', test_metrics)

    def _write_csv(self, name, metrics):
        csv_file = open(os.path.join(self.dst_log, '{}.csv'.format(name)), 'w')
        csv_writer = csv.writer(csv_file)
        header = ['', 'loss', 'acc']
        if FLAGS.labelacc:
            header += [chr(i) for i in range(65, 65 + 26)]
        csv_writer.writerow(header)
        for metric in metrics:
            csv_writer.writerow(metric)
