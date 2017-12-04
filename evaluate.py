import os
import csv

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from dataset import Dataset
from utils import set_embedding_chars
from matplotlib import pyplot as plt

FLAGS = tf.app.flags.FLAGS


class Evaluating():

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._load_dataset()
        self._setup_chars()

    def _setup_dirs(self):
        self.dst_evaluated = os.path.join(FLAGS.gan_dir, 'evaluated')
        if not os.path.exists(self.dst_evaluated):
            os.mkdir(self.dst_evaluated)

    def _load_dataset(self):
        self.generated_dataset = Dataset(FLAGS.generated_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.generated_dataset.set_load_data()
        self.real_dataset = Dataset(FLAGS.font_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.real_dataset.set_load_data()

    def _setup_chars(self):
        self.embedding_chars = set_embedding_chars(FLAGS.embedding_chars_type)
        assert self.embedding_chars != [], 'embedding_chars is empty'
        self.char_embedding_n = len(self.embedding_chars)

    def calc_hamming_distance(self):
        def plot(distances, filename):
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(distances, bins=50, range=(0, 2000))
            plt.savefig(os.path.join(self.dst_evaluated, '{}.png'.format(filename)))
            plt.close()

        min_distances_list = list()
        for c in tqdm(self.embedding_chars):
            generated_n = self.generated_dataset.get_data_n_by_labels([c])
            real_n = self.real_dataset.get_data_n_by_labels([c])
            generated_imgs = np.mean(self.generated_dataset.get_batch_by_labels(0, generated_n, [c]), axis=3)
            real_imgs = np.mean(self.real_dataset.get_batch_by_labels(0, real_n, [c]), axis=3)
            min_distances = list()
            for generated_i in tqdm(range(generated_n)):
                min_distance = float('inf')
                for real_i in range(real_n):
                    distance = np.sum(np.absolute(real_imgs[real_i] - generated_imgs[generated_i]))
                    min_distance = min(min_distance, distance)
                min_distances.append(min_distance)
            plot(min_distances, c)
            min_distances_list.append(min_distances)
        mean_all_min_dinstances = np.mean(np.array(min_distances_list), axis=0).tolist()
        plot(mean_all_min_dinstances, 'all')
        self._write_csv(min_distances_list, mean_all_min_dinstances)

    def _write_csv(self, distances_list, all_distances):
        with open(os.path.join(self.dst_evaluated, 'evaluate.csv'), 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = ['', 'all'] + self.embedding_chars
            csv_writer.writerow(header)
            for i in range(len(all_distances)):
                line = list()
                line.append(i)
                line.append(all_distances[i])
                for j in range(len(distances_list)):
                    line.append(distances_list[j][i])
                csv_writer.writerow(line)
