import os
import sys
import random

import numpy as np
from PIL import Image
import h5py
from glob import glob
from tqdm import tqdm


class Dataset():

    def __init__(self, h5_path, mode, img_width, img_height, img_dim, is_mem=False):
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height
        self.img_dim = img_dim
        self.is_mem = is_mem

        assert mode == 'w' or mode == 'r', 'mode must be \'w\' or \'r\''
        if self.mode == 'w':
            if os.path.exists(h5_path):
                while True:
                    inp = input('overwrite {}? (y/n)\n'.format(h5_path))
                    if inp == 'y' or inp == 'n':
                        break
                if inp == 'n':
                    print('canceled')
                    sys.exit()
            self.h5file = h5py.File(h5_path, mode)
        if self.mode == 'r':
            assert os.path.exists(h5_path), 'hdf5 file is not found: {}'.format(h5_path)
            self.h5file = h5py.File(h5_path, mode)
            if self.is_mem:
                self._get = self._get_from_mem
            else:
                self._get = self._get_from_file

    def load_imgs(self, src_dir_path):
        dir_paths = sorted(glob('{}/*'.format(src_dir_path)))
        for dir_path in tqdm(dir_paths):
            if not os.path.isdir(dir_path):
                continue
            img_paths = sorted(glob('{}/*.png'.format(dir_path)))
            imgs = np.empty((len(img_paths), self.img_width, self.img_height, self.img_dim), dtype=np.float32)
            fontnames = np.empty((len(img_paths)), dtype=object)
            for i, img_path in enumerate(img_paths):
                pil_img = Image.open(img_path)
                np_img = np.asarray(pil_img)
                np_img = (np_img.astype(np.float32) / 127.5) - 1.
                np_img = np_img[np.newaxis, :, :, np.newaxis]
                if self.img_dim == 3:
                    np_img = np.repeat(np_img, 3, axis=3)
                imgs[i] = np_img
                fontnames[i] = os.path.basename(img_path).split('.')[0]
            self._save(os.path.basename(dir_path), imgs, fontnames)

    def _save(self, group_name, imgs, fontnames):
        self.h5file.create_group(group_name)
        self.h5file.create_dataset(group_name + '/imgs', data=imgs)
        self.h5file.create_dataset(group_name + '/fontnames', data=fontnames, dtype=h5py.special_dtype(vlen=str))
        self.h5file.flush()

    def set_load_data(self, train_rate=1.):
        self.keys_queue_train = list()
        is_same_font_n = True
        prev_font_n = 0
        self.label_n = 0
        for key, val in self.h5file.items():
            font_n = len(val['imgs'])
            for i in range(font_n):
                self.keys_queue_train.append((key, i))
            if prev_font_n != 0 and font_n != prev_font_n:
                is_same_font_n = False
            prev_font_n = max(font_n, prev_font_n)
            self.label_n += 1
        self.font_n = prev_font_n
        if train_rate != 1.:
            assert is_same_font_n, 'If you want to divide train/test, all of font num should be same.'
            train_n = int(font_n * train_rate)
            train_ids = random.sample(range(0, font_n), train_n)
            self.keys_queue_test = list(filter(lambda x: x[1] not in train_ids, self.keys_queue_train))
            self.keys_queue_train = list(filter(lambda x: x[1] in train_ids, self.keys_queue_train))
        if self.is_mem:
            self._put_on_mem()

    def shuffle(self, is_test=False):
        if is_test:
            random.shuffle(self.keys_queue_test)
        else:
            random.shuffle(self.keys_queue_train)

    def get_img_len(self, is_test=False):
        if is_test:
            return len(self.keys_queue_test)
        return len(self.keys_queue_train)

    def get_batch(self, batch_i, batch_size, is_test=False, is_label=False):
        keys_list = list()
        for i in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            if is_test:
                keys_list.append(self.keys_queue_test[i])
            else:
                keys_list.append(self.keys_queue_train[i])
        return self._get(keys_list, is_label)

    def get_random(self, batch_size, is_test=False, is_label=False):
        keys_list = list()
        for _ in range(batch_size):
            if is_test:
                keys_list.append(random.choice(self.keys_queue_test))
            else:
                keys_list.append(random.choice(self.keys_queue_train))
        return self._get(keys_list, is_label)

    def get_random_by_labels(self, batch_size, labels, is_test=False, is_label=False):
        if is_test:
            keys_queue = self.keys_queue_test
        else:
            keys_queue = self.keys_queue_train
        filtered_keys_queue = list(filter(lambda x: x[1] in labels, keys_queue))
        keys_list = list()
        for _ in range(batch_size):
            keys_list.append(random.choice(filtered_keys_queue))
        return self._get(keys_list, is_label)

    def _get_from_file(self, keys_list, is_label=False):
        imgs = np.empty((len(keys_list), self.img_width, self.img_height, self.img_dim), np.float32)
        labels = list()
        for i, keys in enumerate(keys_list):
            img = self.h5file[keys[0] + '/imgs'].value[keys[1]]
            imgs[i] = img[np.newaxis, :]
            labels.append(keys[0])
        if is_label:
            return imgs, labels
        return imgs

    def _put_on_mem(self):
        print('putting data on memory...')
        self.imgs = np.empty((self.label_n, self.font_n, self.img_width, self.img_height, self.img_dim), np.float32)
        self.label_to_id = dict()
        self.label_to_font_n = dict()
        for i, key in enumerate(self.h5file.keys()):
            self.imgs[i] = self.h5file[key + '/imgs'].value
            self.label_to_id[key] = i
            self.label_to_font_n[key] = len(self.imgs[i])

    def _get_from_mem(self, keys_list, is_label=False):
        imgs = np.empty((len(keys_list), self.img_width, self.img_height, self.img_dim), np.float32)
        labels = list()
        for i, keys in enumerate(keys_list):
            assert keys[1] < self.label_to_font_n[keys[0]], 'Image is out of range'
            img = self.imgs[self.label_to_id[keys[0]]][keys[1]]
            imgs[i] = img[np.newaxis, :]
            labels.append(keys[0])
        if is_label:
            return imgs, labels
        return imgs
