import os
import csv
from collections import Counter

import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib

from dataset import Dataset
from utils import set_chars_type

FLAGS = tf.app.flags.FLAGS
matplotlib.use('Agg')


class Evaluating():
    """Evaluating generated fonts

    This class is for evaluating generated fonts.
    Measure between generated fonts and real fonts by using pseudo-Hamming distance.
    """

    def __init__(self):
        global FLAGS
        self._setup_dirs()
        self._load_dataset()
        self._setup_chars()

    def _setup_dirs(self):
        """Setup output directories

        If destinations are not existed, make directories like this:
            FLAGS.gan_dir
            └ evaluated
        """
        self.dst_evaluated = os.path.join(FLAGS.gan_dir, 'evaluated')
        if not os.path.exists(self.dst_evaluated):
            os.mkdir(self.dst_evaluated)

    def _load_dataset(self):
        """Load dataset

        Setup dataset, generated fonts and real fonts.
        """
        self.generated_dataset = Dataset(FLAGS.generated_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.generated_dataset.set_load_data()
        self.real_dataset = Dataset(FLAGS.font_h5, 'r', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        self.real_dataset.set_load_data()

    def _setup_chars(self):
        """Setup characters' type

        Setup characters\' type, caps or hiragana or both.
        """
        self.embedding_chars = set_chars_type(FLAGS.chars_type)
        assert self.embedding_chars != [], 'embedding_chars is empty'
        self.char_embedding_n = len(self.embedding_chars)

    def calc_hamming_distance(self):
        """Calculate pseudo-Hamming distance

        Measure between generated fonts and real fonts by using pseudo-Hamming distance.
        If you want to know pseudo-Hamming distance, check this:
        Uchida, et al. "Exploring the World of Fonts for Discovering the Most Standard Fonts and the Missing Fonts", ICDAR, 2015.
        """
        from matplotlib import pyplot as plt

        def transform_backgorund_distance(src_imgs):
            bin_imgs = np.where(src_imgs > 0, 255, 0).astype(np.uint8)
            mask_imgs = np.where(src_imgs > 0, 0, 1).astype(np.uint8)
            dist_imgs = np.empty(src_imgs.shape)
            img_n = src_imgs.shape[0]
            for i in range(img_n):
                dist_imgs[i] = cv2.distanceTransform(bin_imgs[i], cv2.DIST_L2, 3)
            return mask_imgs, dist_imgs

        def plot(distances, filename):
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(1, 1, 1)
            ax.hist(distances, bins=50)
            plt.savefig(os.path.join(self.dst_evaluated, '{}.png'.format(filename)))
            plt.close()

        min_distances_list = list()
        min_real_indices_list = list()
        try:
            for c in tqdm(self.embedding_chars):
                generated_n = self.generated_dataset.get_data_n_by_labels([c])
                generated_imgs = np.mean(self.generated_dataset.get_batch_by_labels(0, generated_n, [c]), axis=3)
                mask_generated_imgs, dist_generated_imgs = transform_backgorund_distance(generated_imgs)
                real_n = self.real_dataset.get_data_n_by_labels([c])
                real_imgs = np.mean(self.real_dataset.get_batch_by_labels(0, real_n, [c]), axis=3)
                mask_real_imgs, dist_real_imgs = transform_backgorund_distance(real_imgs)
                min_distances = list()
                min_real_indices = list()
                for generated_i in tqdm(range(generated_n)):
                    min_distance = float('inf')
                    for real_i in range(real_n):
                        distance = np.sum(np.multiply(dist_generated_imgs[generated_i], mask_real_imgs[real_i]) +
                                          np.multiply(dist_real_imgs[real_i], mask_generated_imgs[generated_i]))
                        if distance < min_distance:
                            min_distance = distance
                            min_real_index = real_i
                    min_distances.append(min_distance)
                    min_real_indices.append(min_real_index)
                plot(min_distances, c)
                min_distances_list.append(min_distances)
                min_real_indices_list.append(min_real_indices)
        except KeyboardInterrupt:
            print('cancelled. but write csv...')
        finally:
            mean_all_min_dinstances = np.mean(np.array(min_distances_list), axis=0).tolist()
            plot(mean_all_min_dinstances, 'all')
            self._write_csv(min_distances_list, mean_all_min_dinstances, min_real_indices_list)

    def _write_csv(self, distances_list, all_distances, real_indices_list):
        """Write CSV

        Write distances in CSV file.

        Args:
            distance_list: A distance, which is only 1 character's.
            all_distances: All character's average.
            real_indices_list: Index list of real fonts, that have minimum distance from a generated font.
        """
        with open(os.path.join(self.dst_evaluated, 'evaluate.csv'), 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            font_n = len(all_distances)
            char_n = len(real_indices_list)
            header = ['', 'all_dist', 'fontname', 'most_n'] + self.embedding_chars[:char_n] + ['index_' + char for char in self.embedding_chars[:char_n]]
            csv_writer.writerow(header)
            for i in range(font_n):
                line = list()
                line.append(i)
                line.append(all_distances[i])
                real_indices = list()
                for j in range(char_n):
                    real_indices.append(real_indices_list[j][i])
                count = Counter(real_indices)
                line.append(self.real_dataset.get_fontname_by_label_id('A', count.most_common()[0][0]))
                line.append(count.most_common()[0][1])
                for j in range(char_n):
                    line.append(distances_list[j][i])
                line += [self.real_dataset.get_fontname_by_label_id('A', real_index) for real_index in real_indices]
                csv_writer.writerow(line)
