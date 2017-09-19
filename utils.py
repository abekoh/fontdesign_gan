import numpy as np
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf


def concat_imgs(src_imgs, row_n, col_n):
    concated_img = np.empty((0, src_imgs.shape[1] * col_n, 1))
    white_img = np.ones((src_imgs.shape[1], src_imgs.shape[2], 1))
    for row_i in range(row_n):
        concated_row_img = np.empty((src_imgs.shape[1], 0, 1))
        for col_i in range(col_n):
            count = row_i * col_n + col_i
            if count < len(src_imgs):
                concated_row_img = np.concatenate((concated_row_img, src_imgs[count]), axis=1)
            else:
                concated_row_img = np.concatenate((concated_row_img, white_img), axis=1)
        concated_img = np.concatenate((concated_img, concated_row_img), axis=0)
    return concated_img


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:
                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)
