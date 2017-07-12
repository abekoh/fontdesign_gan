import os
import numpy as np
from PIL import Image
import h5py
from glob import glob


class Dataset():

    def __init__(self, h5_path, mode):
        self.mode = mode
        self.h5file = h5py.File(h5_path, mode)

    # def __del__(self):
    #     self.h5file.close()

    def load_imgs(self, src_dir_path):
        dir_paths = sorted(glob('{}/*'.format(src_dir_path)))
        for dir_path in dir_paths:
            print('loading {}'.format(dir_path))
            imgs = np.empty((0, 256, 256), dtype=np.float32)
            img_paths = glob('{}/*.png'.format(dir_path))
            for img_path in img_paths:
                pil_img = Image.open(img_path)
                np_img = np.asarray(pil_img)
                np_img = np_img.astype(np.float32) / 255.0
                np_img = np_img[np.newaxis, :]
                imgs = np.append(imgs, np_img, axis=0)
            self._save(dir_path.split('/')[-1], imgs)

    def _save(self, dataname, imgs):
        self.h5file.create_dataset(dataname, data=imgs)
        self.h5file.flush()


if __name__ == '__main__':
    dataset = Dataset('./test.h5', 'w')
    dataset.load_imgs('../../font_dataset/font_jp/')
