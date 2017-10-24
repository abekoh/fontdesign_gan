import numpy as np
from glob import glob
import os
import imageio


def concat_imgs(src_imgs, row_n, col_n):
    concated_img = np.empty((0, src_imgs.shape[1] * col_n, src_imgs.shape[3]))
    white_img = np.ones((src_imgs.shape[1], src_imgs.shape[2], src_imgs.shape[3]))
    for row_i in range(row_n):
        concated_row_img = np.empty((src_imgs.shape[1], 0, src_imgs.shape[3]))
        for col_i in range(col_n):
            count = row_i * col_n + col_i
            if count < len(src_imgs):
                concated_row_img = np.concatenate((concated_row_img, src_imgs[count]), axis=1)
            else:
                concated_row_img = np.concatenate((concated_row_img, white_img), axis=1)
        concated_img = np.concatenate((concated_img, concated_row_img), axis=0)
    return concated_img


def combine_imgs(src_img_list):
    combined_img = np.empty((0, src_img_list[0].shape[1], src_img_list[0].shape[2]))
    for src_img in src_img_list:
        combined_img = np.concatenate((combined_img, src_img), axis=0)
    return combined_img


def make_gif(src_imgs_dir_path, dst_img_path):
    img_paths = glob('{}/*_10.png'.format(src_imgs_dir_path))
    img_filenames = sorted([os.path.basename(path) for path in img_paths],
                           key=lambda x: int(x.split('_')[0]))
    skipped_img_filenames = [img_filenames[i] for i in range(0, 200)]
    sorted_img_paths = [os.path.join(src_imgs_dir_path, filename) for filename in skipped_img_filenames]
    imgs = [imageio.imread(f) for f in sorted_img_paths]
    # for i in range(len(imgs)):
    #     imgs[i] = imgs[i][512:, :, :]
    imageio.mimsave(dst_img_path, imgs)


def diclist_to_list(dicts):
    return {k: v for dic in dicts for k, v in dic.items()}
