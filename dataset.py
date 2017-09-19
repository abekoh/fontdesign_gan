import os
import numpy as np
from PIL import Image
import h5py
from glob import glob
import random


class Dataset():

    def __init__(self, h5_path, mode, img_size=(256, 256), is_binary=False, img_dim=1):
        self.mode = mode
        self.img_size = img_size
        self.is_binary = is_binary
        self.img_dim = img_dim
        self.h5file = h5py.File(h5_path, mode)

    def load_imgs(self, src_dir_path):
        dir_paths = sorted(glob('{}/*'.format(src_dir_path)))
        for dir_path in dir_paths:
            if not os.path.isdir(dir_path):
                continue
            print('loading {}'.format(dir_path))
            imgs = np.empty((0, self.img_size[0], self.img_size[1], self.img_dim), dtype=np.float32)
            img_paths = sorted(glob('{}/*.png'.format(dir_path)))
            labels = np.array([], dtype=object)
            for img_path in img_paths:
                pil_img = Image.open(img_path)
                np_img = np.asarray(pil_img)
                if self.is_binary:
                    np_img = (np_img.astype(np.float32) * 2.) - 1.
                else:
                    np_img = (np_img.astype(np.float32) / 127.5) - 1.
                np_img = np_img[np.newaxis, :, :, np.newaxis]
                if self.img_dim == 3:
                    np_img = np.repeat(np_img, 3, axis=3)
                imgs = np.append(imgs, np_img, axis=0)
                labels = np.append(labels, os.path.basename(img_path).split('.')[0])
            self._save(os.path.basename(dir_path), imgs, labels)

    def _save(self, group_name, imgs, labels):
        self.h5file.create_group(group_name)
        self.h5file.create_dataset(group_name + '/imgs', data=imgs)
        self.h5file.create_dataset(group_name + '/labels', data=labels, dtype=h5py.special_dtype(vlen=str))
        self.h5file.flush()

    def set_load_data(self, train_rate=1.):
        self.keys_queue_train = list()
        for key, value in self.h5file.items():
            for i in range(len(value['labels'])):
                self.keys_queue_train.append((key, i))
        if train_rate != 1.:
            self.keys_queue_test = self.keys_queue_train[int(len(self.keys_queue_train) * train_rate):]
            self.keys_queue_train = self.keys_queue_train[:int(len(self.keys_queue_train) * train_rate)]

    def set_label_ids(self, key=None):
        self.label_ids = dict()
        if key is None:
            max_len = 0
            for k, v in self.h5file.items():
                if len(v) > max_len:
                    key = k
                    max_len = len(v)
        for i, label in enumerate(self.h5file[key + '/labels'].value):
            self.label_ids[label] = i

    def set_category_arange(self):
        self.category_queue = dict()
        for i, key in enumerate(self.h5file.keys()):
            self.category_queue[key] = i

    def set_category_random(self, id_n):
        self.category_queue = dict()
        for key in self.h5file.keys():
            self.category_queue[key] = random.randint(0, id_n - 1)

    def shuffle(self, is_test=False):
        if is_test:
            random.shuffle(self.keys_queue_test)
        else:
            random.shuffle(self.keys_queue_train)

    def get_img_len(self, is_test=False):
        if is_test:
            return len(self.keys_queue_test)
        return len(self.keys_queue_train)

    def get_batch(self, batch_i, batch_size, is_test=False, is_cat=False):
        keys_list = list()
        for i in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            if is_test:
                keys_list.append(self.keys_queue_test[i])
            else:
                keys_list.append(self.keys_queue_train[i])
        return self._get(keys_list, is_cat)

    def get_random(self, batch_size, is_test=False, is_cat=False):
        keys_list = list()
        for i in range(batch_size):
            if is_test:
                keys_list.append(random.choice(self.keys_queue_test))
            else:
                keys_list.append(random.choice(self.keys_queue_train))
        return self._get(keys_list, is_cat)

    def get_selected(self, labels, is_test=False, is_cat=False):
        keys_list = list()
        for label in labels:
            num = self.get_label_id(label)
            if is_test:
                keys_list.append(self.keys_queue_test[num])
            else:
                keys_list.append(self.keys_queue_train[num])
        return self._get(keys_list, is_cat)

    def get_all(self, is_test=False, is_cat=False):
        if is_test:
            return self.get_batch(0, len(self.key_queue_test), is_test)
        return self.get_batch(0, len(self.keys_queue_train), is_test, is_cat)

    def _get(self, keys_list, is_cat=False):
        imgs = np.empty((0, self.img_size[0], self.img_size[1], 1), np.float32)
        labels = list()
        cats = list()
        for keys in keys_list:
            img = self.h5file[keys[0] + '/imgs'].value[keys[1]]
            img = img[np.newaxis, :]
            imgs = np.append(imgs, img, axis=0)
            labels.append(self.h5file[keys[0] + '/labels'].value[keys[1]])
            if is_cat:
                cats.append(self.category_queue[keys[0]])
        if is_cat:
            return imgs, labels, cats
        return imgs, labels

    def get_label_id(self, label):
        return self.label_ids[label]
