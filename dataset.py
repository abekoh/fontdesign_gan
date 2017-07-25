import os
import numpy as np
from PIL import Image
import h5py
from glob import glob
import random


class Dataset():

    def __init__(self, h5_path, mode):
        self.mode = mode
        self.h5file = h5py.File(h5_path, mode)

    def __del__(self):
        self.h5file.close()

    def load_imgs(self, src_dir_path):
        dir_paths = sorted(glob('{}/*'.format(src_dir_path)))
        for dir_path in dir_paths:
            print('loading {}'.format(dir_path))
            imgs = np.empty((0, 256, 256, 1), dtype=np.float32)
            img_paths = sorted(glob('{}/*.png'.format(dir_path)))
            labels = np.array([], dtype=object)
            for img_path in img_paths:
                pil_img = Image.open(img_path)
                np_img = np.asarray(pil_img)
                np_img = np_img.astype(np.float32) / 255.0
                np_img = np_img[np.newaxis, :, :, np.newaxis]
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
            for i in range(value['labels'].len()):
                self.keys_queue_train.append((key, i))
        if train_rate != 1.:
            self.keys_queue_test = self.keys_queue_train[int(len(self.keys_queue_train) * train_rate):]
            self.keys_queue_train = self.keys_queue_train[:int(len(self.keys_queue_train) * train_rate)]

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

    def _get(self, keys_list):
        imgs = np.empty((0, 256, 256, 1), np.float32)
        labels = list()
        for keys in keys_list:
            img = self.h5file[keys[0] + '/imgs'].value[keys[1]]
            img = img[np.newaxis, :]
            imgs = np.append(imgs, img, axis=0)
            labels.append(self.h5file[keys[0] + '/labels'].value[keys[1]])
        return imgs, labels


if __name__ == '__main__':
    dataset = Dataset('src/fonts_200_caps_256x256', 'w')
    dataset.load_imgs('../../font_dataset/png/200_256x256')
    # dataset = Dataset('./test.h5', 'r')
    # dataset.set_load_data()
    # for i in range(2):
    #     imgs, labels = dataset.get_batch(i, 10)
