import os
import sys
import random
import math

import numpy as np
from PIL import Image
import h5py
from glob import glob
from tqdm import tqdm

from utils import concat_imgs


class Dataset():

    def __init__(self, h5_path, mode, img_width, img_height, img_dim, is_mem=False):
        self.mode = mode
        self.img_width = img_width
        self.img_height = img_height
        self.img_dim = img_dim
        self.is_mem = is_mem

        assert mode == 'r' or mode == 'w', 'mode must be \'r\' or \'w\''
        if self.mode == 'r':
            assert os.path.exists(h5_path), 'hdf5 file is not found: {}'.format(h5_path)
            self.h5file = h5py.File(h5_path, mode)
            # # TODO: be more clearly
            # keys = self.h5file.keys()
            # self.data_n = len(keys)
            # for key in keys:
            #     self.label_n = len(self.h5file[key + '/labels'].value)
            #     break
            # if self.is_mem:
            #     self._put_on_mem()
            #     self._get = self._get_from_mem
            # else:
            #     self._get = self._get_from_file
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
        # if self.is_mem:
        #     iters = enumerate(self.h5file.values())
        # else:
        #     iters = self.h5file.items()
        for key, val in self.h5file.items():
            for i in range(len(val['imgs'])):
                self.keys_queue_train.append((key, i))
        if train_rate != 1.:
            self.keys_queue_test = self.keys_queue_train[int(len(self.keys_queue_train) * train_rate):]
            self.keys_queue_train = self.keys_queue_train[:int(len(self.keys_queue_train) * train_rate)]
        print(self.keys_queue_test)

    # def _set_label_ids(self, key=None):
    #     self.label_ids = dict()
    #     if key is None:
    #         max_len = 0
    #         for k, v in self.h5file.items():
    #             if len(v) > max_len:
    #                 key = k
    #                 max_len = len(v)
    #     for i, label in enumerate(self.h5file[key + '/labels'].value):
    #         self.label_ids[label] = i

    def shuffle(self, is_test=False):
        if is_test:
            random.shuffle(self.keys_queue_test)
        else:
            random.shuffle(self.keys_queue_train)

    def get_img_len(self, is_test=False):
        if is_test:
            return len(self.keys_queue_test)
        return len(self.keys_queue_train)

    def get_batch(self, batch_i, batch_size, is_test=False, is_label=True):
        keys_list = list()
        for i in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            if is_test:
                keys_list.append(self.keys_queue_test[i])
            else:
                keys_list.append(self.keys_queue_train[i])
        return self._get(keys_list, is_label)

    def get_random(self, batch_size, is_test=False, is_label=True):
        keys_list = list()
        for _ in range(batch_size):
            if is_test:
                keys_list.append(random.choice(self.keys_queue_test))
            else:
                keys_list.append(random.choice(self.keys_queue_train))
        return self._get(keys_list, is_label)

    # def get_selected(self, labels, is_test=False, is_label=True):
    #     keys_list = list()
    #     for label in labels:
    #         num = self.get_label_id(label)
    #         if is_test:
    #             keys_list.append(self.keys_queue_test[num])
    #         else:
    #             keys_list.append(self.keys_queue_train[num])
    #     return self._get(keys_list, is_label)

    def get_random_by_ids(self, batch_size, ids, is_test=False, is_label=True):
        if is_test:
            keys_queue = self.keys_queue_test
        else:
            keys_queue = self.keys_queue_train
        filtered_keys_queue = list(filter(lambda x: x[1] in ids, keys_queue))
        keys_list = list()
        for _ in range(batch_size):
            keys_list.append(random.choice(filtered_keys_queue))
        return self._get(keys_list, is_label)

    def get_all(self, is_test=False, is_label=True):
        if is_test:
            return self.get_batch(0, len(self.key_queue_test), is_test)
        return self.get_batch(0, len(self.keys_queue_train), is_test, is_label)

    def _get_from_file(self, keys_list, is_label=True):
        imgs = np.empty((len(keys_list), self.img_width, self.img_height, self.img_dim), np.float32)
        labels = list()
        for i, keys in enumerate(keys_list):
            img = self.h5file[keys[0] + '/imgs'].value[keys[1]]
            imgs[i] = img[np.newaxis, :]
            labels.append(self.h5file[keys[0] + '/labels'].value[keys[1]])
        if is_label:
            return imgs, labels
        return imgs

    def _put_on_mem(self):
        print('putting data on memory...')
        self.imgs = np.empty((self.data_n, self.label_n, self.img_width, self.img_height, self.img_dim), np.float32)
        self.labels = np.empty((self.data_n, self.label_n), object)
        for i, key in enumerate(self.h5file.keys()):
            self.imgs[i] = self.h5file[key + '/imgs'].value
            self.labels[i] = self.h5file[key + '/labels'].value

    def _get_from_mem(self, keys_list, is_label=True):
        imgs = np.empty((len(keys_list), self.img_width, self.img_height, self.img_dim), np.float32)
        labels = list()
        for i, keys in enumerate(keys_list):
            img = self.imgs[keys[0]][keys[1]]
            imgs[i] = img[np.newaxis, :]
            labels.append(self.labels[keys[0]][keys[1]])
        if is_label:
            return imgs, labels
        return imgs

    # def _get_label_id(self, label):
    #     return self.label_ids[label]

    def show_random(self):
        imgs, _ = self.get_random(64)
        concated_img = concat_imgs(imgs, 8, 8)
        concated_img = (concated_img + 1.) * 127.5
        concated_img = np.reshape(concated_img, (self.img_width * 8, -1))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.show()

    def save_index(self, dst_img_path):
        imgs_n = len(self.keys_queue_train)
        imgs = self.get_all(is_label=False)
        col_n = math.ceil(math.sqrt(imgs_n))
        concated_img = concat_imgs(imgs, col_n, col_n)
        concated_img = (concated_img + 1.) * 127.5
        concated_img = np.reshape(concated_img, (-1, col_n * self.img_width, self.img_dim))
        pil_img = Image.fromarray(np.uint8(concated_img))
        pil_img.save(dst_img_path)


if __name__ == '__main__':
    dataset = Dataset('./src/200_64x64.h5', 'r', 64, 64, 3, is_mem=True)
    # dataset.load_imgs('../../font_dataset/200_64x64_by_char')
    dataset.set_load_data(0.9)
