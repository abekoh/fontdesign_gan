from keras.models import Model
from keras.layers import Input, Activation, Dropout, Embedding, Reshape, Flatten, Dense, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.initializers import random_normal, truncated_normal


def Generator():
    # Encoder
    en_inp = Input(shape=(256, 256, 1))

    en_1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                  kernel_initializer=truncated_normal(stddev=0.02), name='en_1')(en_inp)
    # -> (:, 128, 128, 64)

    en_2 = LeakyReLU(alpha=0.2)(en_1)
    en_2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_2)
    en_2 = BatchNormalization(momentum=0.9, epsilon=0.00001, name='en_2')(en_2)
    # -> (:, 64, 64, 128)

    en_3 = LeakyReLU(alpha=0.2)(en_2)
    en_3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_3)
    en_3 = BatchNormalization(momentum=0.9, epsilon=0.00001, name='en_3')(en_3)
    # -> (:, 32, 32, 256)

    en_4 = LeakyReLU(alpha=0.2)(en_3)
    en_4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_4)
    en_4 = BatchNormalization(momentum=0.9, epsilon=0.00001, name='en_4')(en_4)
    # -> (:, 16, 16, 512)

    en_5 = LeakyReLU(alpha=0.2)(en_4)
    en_5 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_5)
    en_5 = BatchNormalization(momentum=0.9, epsilon=0.00001, name='en_5')(en_5)
    # -> (:, 8, 8, 512)

    en_6 = LeakyReLU(alpha=0.2)(en_5)
    en_6 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_6)
    en_6 = BatchNormalization(momentum=0.9, epsilon=0.00001, name='en_6')(en_6)
    # -> (:, 4, 4, 512)

    en_7 = LeakyReLU(alpha=0.2)(en_6)
    en_7 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_7)
    en_7 = BatchNormalization(momentum=0.9, epsilon=0.00001, name='en_7')(en_7)
    # -> (:, 2, 2, 512)

    en_8 = LeakyReLU(alpha=0.2)(en_7)
    en_8 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_8)
    en_8 = BatchNormalization(momentum=0.9, epsilon=0.00001, name='en_8')(en_8)
    # -> (:, 1, 1, 512)

    # Embedding
    embedding_inp = Input(shape=(1,), dtype='int32')
    # -> (:)
    embedding = Embedding(40, 128, embeddings_initializer=random_normal(stddev=0.01), name='embedding')(embedding_inp)
    # -> (:, 1, 128)
    embedding = Reshape((1, 1, 128))(embedding)
    # -> (:, 1, 1, 128)

    # Decoder
    de_inp = concatenate([en_8, embedding], axis=3)

    de_1 = Activation('relu')(de_inp)
    de_1 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_1)
    de_1 = BatchNormalization(momentum=0.9, epsilon=0.00001)(de_1)
    de_1 = Dropout(0.5)(de_1)
    # -> (:, 2, 2, 512)
    de_1 = concatenate([de_1, en_7], axis=3)
    # -> (:, 2, 2, 1024)

    de_2 = Activation('relu')(de_1)
    de_2 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_2)
    de_2 = BatchNormalization(momentum=0.9, epsilon=0.00001)(de_2)
    de_2 = Dropout(0.5)(de_2)
    # -> (:, 4, 4, 512)
    de_2 = concatenate([de_2, en_6], axis=3)
    # -> (:, 4, 4, 1024)

    de_3 = Activation('relu')(de_2)
    de_3 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_3)
    de_3 = BatchNormalization(momentum=0.9, epsilon=0.00001)(de_3)
    de_3 = Dropout(0.5)(de_3)
    # -> (:, 8, 8, 512)
    de_3 = concatenate([de_3, en_5], axis=3)
    # -> (:, 8, 8, 1024)

    de_4 = Activation('relu')(de_3)
    de_4 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_4)
    de_4 = BatchNormalization(momentum=0.9, epsilon=0.00001)(de_4)
    # -> (:, 16, 16, 512)
    de_4 = concatenate([de_4, en_4], axis=3)
    # -> (:, 16, 16, 1024)

    de_5 = Activation('relu')(de_4)
    de_5 = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_5)
    de_5 = BatchNormalization(momentum=0.9, epsilon=0.00001)(de_5)
    # -> (:, 32, 32, 256)
    de_5 = concatenate([de_5, en_3], axis=3)
    # -> (:, 32, 32, 512)

    de_6 = Activation('relu')(de_5)
    de_6 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_6)
    de_6 = BatchNormalization(momentum=0.9, epsilon=0.00001)(de_6)
    # -> (:, 64, 64, 128)
    de_6 = concatenate([de_6, en_2], axis=3)
    # -> (:, 64, 64, 256)

    de_7 = Activation('relu')(de_6)
    de_7 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_7)
    de_7 = BatchNormalization(momentum=0.9, epsilon=0.00001)(de_7)
    # -> (:, 128, 128, 64)
    de_6 = concatenate([de_7, en_1], axis=3)
    # -> (:, 128, 128, 128)

    de_8 = Activation('relu')(de_7)
    de_8 = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_8)
    de_8 = Activation('sigmoid')(de_8)
    # -> (:, 256, 256, 1)

    model = Model(inputs=[en_inp, embedding_inp], outputs=de_8)

    return model


def Discriminator():
    dis_inp = Input(shape=(256, 256, 1))

    dis_1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(dis_inp)
    dis_1 = LeakyReLU(alpha=0.2)(dis_1)
    # -> (:, 128, 128, 64)

    dis_2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(dis_1)
    # dis_2 = BatchNormalization(momentum=0.9, epsilon=0.00001)(dis_2)
    dis_2 = LeakyReLU(alpha=0.2)(dis_2)
    # -> (:, 64, 64, 128)

    dis_3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(dis_2)
    # dis_3 = BatchNormalization(momentum=0.9, epsilon=0.00001)(dis_3)
    dis_3 = LeakyReLU(alpha=0.2)(dis_3)
    # -> (:, 32, 32, 256)

    dis_4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(dis_3)
    # dis_4 = BatchNormalization(momentum=0.9, epsilon=0.00001)(dis_4)
    dis_4 = LeakyReLU(alpha=0.2)(dis_4)
    # -> (:, 16, 16, 512)

    fc_0 = Flatten()(dis_4)
    fc_1 = Dense(1, activation='sigmoid')(fc_0)

    fc_2 = Dense(40, activation='softmax')(fc_0)

    model = Model(inputs=dis_inp, outputs=[fc_1, fc_2])

    return model
