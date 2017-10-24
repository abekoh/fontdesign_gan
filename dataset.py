import os
import numpy as np
from PIL import Image
import h5py
from glob import glob
import random

from utils import concat_imgs


class Dataset():

    def __init__(self, h5_path, mode, img_size=(256, 256), is_binary=False, is_mem=False, img_dim=1):
        self.mode = mode
        self.img_size = img_size
        self.is_binary = is_binary
        self.is_mem = is_mem
        self.img_dim = img_dim
        self.h5file = h5py.File(h5_path, mode)
        self._get = self._get_from_file
        if self.mode == 'r':
            # TODO: be more clearly
            keys = self.h5file.keys()
            self.data_n = len(keys)
            for key in keys:
                self.label_n = len(self.h5file[key + '/labels'].value)
                break
            if self.is_mem:
                self._put_on_mem()
                # self._set_label_ids()
                self._get = self._get_from_mem

    def load_imgs(self, src_dir_path):
        dir_paths = sorted(glob('{}/*'.format(src_dir_path)))
        for dir_path in dir_paths:
            if not os.path.isdir(dir_path):
                continue
            print('loading {}'.format(dir_path))
            img_paths = sorted(glob('{}/*.png'.format(dir_path)))
            if len(img_paths) == 0:
                print('.png images are not found')
                continue
            imgs = np.empty((len(img_paths), self.img_size[0], self.img_size[1], self.img_dim), dtype=np.float32)
            labels = np.empty((len(img_paths)), dtype=object)
            for i, img_path in enumerate(img_paths):
                pil_img = Image.open(img_path)
                np_img = np.asarray(pil_img)
                if self.is_binary:
                    np_img = (np_img.astype(np.float32) * 2.) - 1.
                else:
                    np_img = (np_img.astype(np.float32) / 127.5) - 1.
                np_img = np_img[np.newaxis, :, :, np.newaxis]
                if self.img_dim == 3:
                    np_img = np.repeat(np_img, 3, axis=3)
                imgs[i] = np_img
                labels[i] = os.path.basename(img_path).split('.')[0]
            self._save(os.path.basename(dir_path), imgs, labels)

    def _save(self, group_name, imgs, labels):
        self.h5file.create_group(group_name)
        self.h5file.create_dataset(group_name + '/imgs', data=imgs)
        self.h5file.create_dataset(group_name + '/labels', data=labels, dtype=h5py.special_dtype(vlen=str))
        self.h5file.flush()

    def set_load_data(self, train_rate=1.):
        self.keys_queue_train = list()
        if self.is_mem:
            iters = enumerate(self.h5file.values())
        else:
            iters = self.h5file.items()
        for x, value in iters:
            for i in range(len(value['labels'])):
                self.keys_queue_train.append((x, i))
        if train_rate != 1.:
            self.keys_queue_test = self.keys_queue_train[int(len(self.keys_queue_train) * train_rate):]
            self.keys_queue_train = self.keys_queue_train[:int(len(self.keys_queue_train) * train_rate)]

    def _set_label_ids(self, key=None):
        self.label_ids = dict()
        if key is None:
            max_len = 0
            for k, v in self.h5file.items():
                if len(v) > max_len:
                    key = k
                    max_len = len(v)
        for i, label in enumerate(self.h5file[key + '/labels'].value):
            self.label_ids[label] = i

    def shuffle(self, is_test=False):
        if is_test:
            random.shuffle(self.keys_queue_test)
        else:
            random.shuffle(self.keys_queue_train)

    def get_img_len(self, is_test=False):
        if is_test:
            return len(self.keys_queue_test)
        return len(self.keys_queue_train)

    def get_batch(self, batch_i, batch_size, is_test=False):
        keys_list = list()
        for i in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            if is_test:
                keys_list.append(self.keys_queue_test[i])
            else:
                keys_list.append(self.keys_queue_train[i])
        return self._get(keys_list)

    def get_random(self, batch_size, is_test=False):
        keys_list = list()
        for i in range(batch_size):
            if is_test:
                keys_list.append(random.choice(self.keys_queue_test))
            else:
                keys_list.append(random.choice(self.keys_queue_train))
        return self._get(keys_list)

    def get_selected(self, labels, is_test=False):
        keys_list = list()
        for label in labels:
            num = self.get_label_id(label)
            if is_test:
                keys_list.append(self.keys_queue_test[num])
            else:
                keys_list.append(self.keys_queue_train[num])
        return self._get(keys_list)

    def get_all(self, is_test=False):
        if is_test:
            return self.get_batch(0, len(self.key_queue_test), is_test)
        return self.get_batch(0, len(self.keys_queue_train), is_test)

    def _get_from_file(self, keys_list):
        imgs = np.empty((len(keys_list), self.img_size[0], self.img_size[1], self.img_dim), np.float32)
        labels = list()
        for i, keys in enumerate(keys_list):
            img = self.h5file[keys[0] + '/imgs'].value[keys[1]]
            imgs[i] = img[np.newaxis, :]
            labels.append(self.h5file[keys[0] + '/labels'].value[keys[1]])
        return imgs, labels

    def _put_on_mem(self):
        print('putting data on memory...')
        self.imgs = np.empty((self.data_n, self.label_n, self.img_size[0], self.img_size[1], self.img_dim), np.float32)
        self.labels = np.empty((self.data_n, self.label_n), object)
        for i, key in enumerate(self.h5file.keys()):
            self.imgs[i] = self.h5file[key + '/imgs'].value
            self.labels[i] = self.h5file[key + '/labels'].value

    def _get_from_mem(self, keys_list):
        imgs = np.empty((len(keys_list), self.img_size[0], self.img_size[1], self.img_dim), np.float32)
        labels = list()
        for i, keys in enumerate(keys_list):
            img = self.imgs[keys[0]][keys[1]]
            imgs[i] = img[np.newaxis, :]
            labels.append(self.labels[keys[0]][keys[1]])
        return imgs, labels

    def _get_label_id(self, label):
        return self.label_ids[label]

    def show_random(self):
        imgs, _ = self.get_random(64)
        concated_img = concat_imgs(imgs, 8, 8)
        concated_img = (concated_img + 1.) * 127.5
        concated_img = np.reshape(concated_img, (self.img_size[0] * 8, -1))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.show()
