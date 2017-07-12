import os
import numpy as np
from PIL import Image
import h5py
import json
from glob import glob


class Dataset():

    def __init__(self, lang='eng_caps'):
        self.src_imgs = np.empty((0, 256, 256, 1), np.float32)
        self.dst_imgs = np.empty((0, 256, 256, 1), np.float32)
        self.char_ids = np.array([])
        self.font_ids = np.array([])

    def load_images(self, fonts_dir_path, src_font_name, lang='eng_caps'):
        self.charset = self._load_charset(lang=lang, src_json_path='./cjk.json')
        font_dir_paths = self._get_dir_paths(fonts_dir_path)
        src_font_dir_path = os.path.join(fonts_dir_path, src_font_name)
        font_id = 0
        for i, font_dir_path in enumerate(font_dir_paths):
            print('loading ({}/{}): {}'.format(i + 1, len(font_dir_paths), font_dir_path))
            if font_dir_path == src_font_dir_path:
                print('this is src')
                self._load_font_imgs(font_dir_path, True)
            else:
                self._load_font_imgs(font_dir_path, False, font_id=font_id)
                font_id += 1

    def _load_charset(self, lang, src_json_path=''):
        if lang == 'eng_caps':
            caps = [chr(i) for i in range(65, 65 + 26)]
            return caps
        cjk = json.load(open(src_json_path))
        return cjk[lang]

    def _get_dir_paths(self, dir_path):
        dir_paths = []
        for filename in glob(os.path.join(dir_path, '*')):
            if os.path.isdir(filename):
                dir_paths.append(filename)
        dir_paths.sort()
        return dir_paths

    def _load_font_imgs(self, font_dir_path, is_src, font_id):
        for char_id, c in enumerate(self.charset):
            img_pil = Image.open(os.path.join(font_dir_path, '{}.png'.format(c)))
            img_np = np.asarray(img_pil)
            img_np = img_np.astype(np.float32) / 255.0
            img_np = img_np[np.newaxis, :, :, np.newaxis]
            if is_src:
                self.src_imgs = np.append(self.src_imgs, img_np, axis=0)
            else:
                self.dst_imgs = np.append(self.dst_imgs, img_np, axis=0)
                self.char_ids = np.append(self.char_ids, np.array([char_id]))
                self.font_ids = np.append(self.font_ids, np.array([font_id]))

    def make_train_test(self, train_n_rate=0.9):
        combined = np.c_[self.dst_imgs.reshape(self.dst_imgs.shape[0], -1), self.char_ids]
        np.random.shuffle(combined)
        imgs_n = self.dst_imgs.size // self.dst_imgs.shape[0]
        imgs = combined[:, :imgs_n].reshape(self.dst_imgs.shape)
        labels = combined[:, imgs_n:].reshape(self.char_ids.shape)
        train_n = int(imgs.shape[0] * train_n_rate)
        self.train_imgs = imgs[:train_n]
        self.train_labels = labels[:train_n]
        self.test_imgs = imgs[train_n:]
        self.test_labels = labels[train_n:]

    def save_h5(self, dst_hdf5_path):
        h5file = h5py.File(dst_hdf5_path, 'w')
        h5file.create_dataset('src_imgs', data=self.src_imgs)
        h5file.create_dataset('dst_imgs', data=self.dst_imgs)
        h5file.create_dataset('char_ids', data=self.char_ids)
        h5file.create_dataset('font_ids', data=self.font_ids)
        h5file.flush()
        h5file.close()

    def save_h5_for_classifier(self, dst_hdf5_path):
        h5file = h5py.File(dst_hdf5_path, 'w')
        h5file.create_dataset('train_imgs', data=self.train_imgs)
        h5file.create_dataset('test_imgs', data=self.test_imgs)
        h5file.create_dataset('train_labels', data=self.train_labels)
        h5file.create_dataset('test_labels', data=self.test_labels)
        h5file.flush()
        h5file.close()

    def load_h5(self, src_hdf5_path):
        h5file = h5py.File(src_hdf5_path, 'r')
        self.src_imgs = h5file['src_imgs'].value
        self.dst_imgs = h5file['dst_imgs'].value
        self.char_ids = h5file['char_ids'].value
        self.font_ids = h5file['font_ids'].value
        h5file.close()

    def load_h5_for_classifier(self, src_hdf5_path):
        h5file = h5py.File(src_hdf5_path, 'r')
        self.train_imgs = h5file['train_imgs'].value
        self.train_labels = h5file['train_labels'].value
        self.test_imgs = h5file['test_imgs'].value
        self.test_labels = h5file['test_labels'].value
        h5file.close()

    def get(self):
        return self.src_imgs, self.dst_imgs, self.char_ids, self.font_ids

    def get_for_classifier(self):
        return self.train_imgs, self.train_labels, self.test_imgs, self.test_labels


if __name__ == '__main__':
    # dataset = Dataset(lang='eng_caps')
    # dataset.load_images('../../font_dataset/selected_200_256x256', 'Arial', lang='jp')
    # dataset.save_h5('./font_200_selected_alphs.h5')
    dataset = Dataset(lang='eng_caps')
    dataset.load_h5('./font_200_selected_alphs.h5')
    dataset.make_train_test()
    dataset.save_h5_for_classifier('./font_200_selected_alphs_for_classifier.h5')
