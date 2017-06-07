from keras.models import Model
from keras.layers import Input, Activation, Dropout, Embedding, Reshape, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.initializers import random_normal


def Generator():
    # Encoder
    en_inp = Input(shape=(256, 256, 3))

    en_1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='en_1')(en_inp)
    # -> (:, 128, 128, 64)

    en_2 = LeakyReLU(alpha=0.2)(en_1)
    en_2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(en_2)
    en_2 = BatchNormalization(name='en_2')(en_2)
    # -> (:, 64, 64, 128)

    en_3 = LeakyReLU(alpha=0.2)(en_2)
    en_3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(en_3)
    en_3 = BatchNormalization(name='en_3')(en_3)
    # -> (:, 32, 32, 256)

    en_4 = LeakyReLU(alpha=0.2)(en_3)
    en_4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(en_4)
    en_4 = BatchNormalization(name='en_4')(en_4)
    # -> (:, 16, 16, 512)

    en_5 = LeakyReLU(alpha=0.2)(en_4)
    en_5 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(en_5)
    en_5 = BatchNormalization(name='en_5')(en_5)
    # -> (:, 8, 8, 512)

    en_6 = LeakyReLU(alpha=0.2)(en_5)
    en_6 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(en_6)
    en_6 = BatchNormalization(name='en_6')(en_6)
    # -> (:, 4, 4, 512)

    en_7 = LeakyReLU(alpha=0.2)(en_6)
    en_7 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(en_7)
    en_7 = BatchNormalization(name='en_7')(en_7)
    # -> (:, 2, 2, 512)

    en_8 = LeakyReLU(alpha=0.2)(en_7)
    en_8 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(en_8)
    en_8 = BatchNormalization(name='en_8')(en_8)
    # -> (:, 1, 1, 512)

    # Embedding
    embedding_inp = Input(shape=(1,), dtype='int32')

    embedding = Embedding(40, 128, embeddings_initializer=random_normal(stddev=0.01), name='embedding')(embedding_inp)
    embedding = Reshape((1, 1, 128))(embedding)

    de_inp = concatenate([en_8, embedding], axis=3)

    de_1 = Activation('relu')(de_inp)
    de_1 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(de_1)
    de_1 = BatchNormalization()(de_1)
    de_1 = Dropout(0.5)(de_1)
    # -> (:, 2, 512, 512)
    de_1 = concatenate([de_1, en_7], axis=3)

    de_2 = Activation('relu')(de_1)
    de_2 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(de_2)
    de_2 = BatchNormalization()(de_2)
    de_2 = Dropout(0.5)(de_2)
    de_2 = concatenate([de_2, en_6], axis=3)

    de_3 = Activation('relu')(de_2)
    de_3 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(de_3)
    de_3 = BatchNormalization()(de_3)
    de_3 = Dropout(0.5)(de_3)
    de_3 = concatenate([de_3, en_5], axis=3)

    de_4 = Activation('relu')(de_3)
    de_4 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(de_4)
    de_4 = BatchNormalization()(de_4)
    de_4 = concatenate([de_4, en_4], axis=3)

    de_5 = Activation('relu')(de_4)
    de_5 = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(de_5)
    de_5 = BatchNormalization()(de_5)
    de_5 = concatenate([de_5, en_3], axis=3)

    de_6 = Activation('relu')(de_5)
    de_6 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(de_6)
    de_6 = BatchNormalization()(de_6)
    de_6 = concatenate([de_6, en_2], axis=3)

    de_7 = Activation('relu')(de_6)
    de_7 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(de_7)
    de_7 = BatchNormalization()(de_7)
    de_6 = concatenate([de_7, en_1], axis=3)

    de_8 = Activation('relu')(de_7)
    de_8 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(de_8)

    model = Model(inputs=[en_inp, embedding_inp], outputs=de_8)

    return model
