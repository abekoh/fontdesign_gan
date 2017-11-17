import os
from glob import glob
import math

import numpy as np
from matplotlib import pyplot as plt
import imageio

ALPHABET_CAPS = list(chr(i) for i in range(65, 65 + 26))
HIRAGANA_SEION = list('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐゑをん')


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


def divide_img_dims(src_imgs):
    divided_imgs = np.empty((src_imgs.shape[2], src_imgs.shape[0], src_imgs.shape[1]))
    for dim_i in range(src_imgs.shape[2]):
        divided_img = src_imgs[:, :, dim_i]
        divided_imgs[dim_i] = divided_img
    return divided_imgs


def combine_imgs(src_img_list):
    combined_img = np.empty((0, src_img_list[0].shape[1], src_img_list[0].shape[2]))
    for src_img in src_img_list:
        combined_img = np.concatenate((combined_img, src_img), axis=0)
    return combined_img


def save_heatmap(imgs, title, dst_path, vmin=None, vmax=None):
    print('making heatmap... ({})'.format(title))
    edge_n = get_imgs_edge_n(imgs.shape[0])
    if not vmin:
        vmin = np.min(imgs)
    if not vmax:
        vmax = np.max(imgs)
    fig, axes = plt.subplots(edge_n, edge_n, figsize=(10, 10))
    for i in range(edge_n ** 2):
        row_i = i // edge_n
        col_i = i % edge_n
        if i >= imgs.shape[0]:
            axes[row_i, col_i].axis('off')
            continue
        axes[row_i, col_i].pcolor(imgs[i], vmin=vmin, vmax=vmax, cmap=plt.get_cmap('plasma'))
        axes[row_i, col_i].set_aspect('equal', 'box')
        axes[row_i, col_i].tick_params(labelbottom='off', bottom='off', labelleft='off', left='off')
        axes[row_i, col_i].set_xticklabels([])
    plt.suptitle(title)
    plt.savefig(dst_path, dpi=100)
    plt.close()


def save_bargraph(arr, title, dst_path, vmin=None, vmax=None):
    edge_n = get_imgs_edge_n(arr.shape[0])
    if not vmin:
        vmin = np.min(arr)
    if not vmax:
        vmax = np.max(arr)
    if arr.shape[0] < edge_n ** 2:
        arr = np.concatenate((arr, np.zeros((edge_n ** 2 - arr.shape[0]))))
    fig, axes = plt.subplots(edge_n, figsize=(10, 10))
    for i in range(edge_n):
        first = i * edge_n
        last = (i + 1) * edge_n
        index = np.arange(first, last)
        axes[i].bar(index, arr[first:last])
        axes[i].set_ylim([vmin, vmax])
        axes[i].tick_params(labelbottom='off', bottom='off')
    plt.suptitle(title)
    plt.savefig(dst_path, dpi=100)
    plt.close()


def get_imgs_edge_n(img_n):
    return math.ceil(math.sqrt(img_n))


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


def set_embedding_chars(embedding_chars_type):
    embedding_chars = list()
    if 'caps' in embedding_chars_type:
        embedding_chars.extend(ALPHABET_CAPS)
    if 'hiragana' in embedding_chars_type:
        embedding_chars.extend(HIRAGANA_SEION)
    return embedding_chars
