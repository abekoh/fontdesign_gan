from keras.models import Model
from keras.layers import Input, Activation, Dropout, Embedding, Reshape, Flatten, Dense, concatenate, MaxPool2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.initializers import random_normal, truncated_normal
from ops import Subtract


def GeneratorPix2Pix(img_size=(256, 256), img_dim=1, font_embedding_n=40):
    # Encoder
    en_inp = Input(shape=(img_size[0], img_size[1], img_dim))
    # -> (:, 256, 256, img_dim)

    en_1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_inp)
    # -> (:, 128, 128, 64)

    en_2 = LeakyReLU(alpha=0.2)(en_1)
    en_2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_2)
    en_2 = BatchNormalization(momentum=0.9, epsilon=0.00001)(en_2)
    # -> (:, 64, 64, 128)

    en_3 = LeakyReLU(alpha=0.2)(en_2)
    en_3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_3)
    en_3 = BatchNormalization(momentum=0.9, epsilon=0.00001)(en_3)
    # -> (:, 32, 32, 256)

    en_4 = LeakyReLU(alpha=0.2)(en_3)
    en_4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_4)
    en_4 = BatchNormalization(momentum=0.9, epsilon=0.00001)(en_4)
    # -> (:, 16, 16, 512)

    en_5 = LeakyReLU(alpha=0.2)(en_4)
    en_5 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_5)
    en_5 = BatchNormalization(momentum=0.9, epsilon=0.00001)(en_5)
    # -> (:, 8, 8, 512)

    en_6 = LeakyReLU(alpha=0.2)(en_5)
    en_6 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_6)
    en_6 = BatchNormalization(momentum=0.9, epsilon=0.00001)(en_6)
    # -> (:, 4, 4, 512)

    en_7 = LeakyReLU(alpha=0.2)(en_6)
    en_7 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_7)
    en_7 = BatchNormalization(momentum=0.9, epsilon=0.00001)(en_7)
    # -> (:, 2, 2, 512)

    en_8 = LeakyReLU(alpha=0.2)(en_7)
    en_8 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(en_8)
    en_8 = BatchNormalization(momentum=0.9, epsilon=0.00001, name='en_last')(en_8)
    # -> (:, 1, 1, 512)

    # Embedding
    embedding_inp = Input(shape=(1,), dtype='int32')
    # -> (:)
    embedding = Embedding(font_embedding_n, 128, embeddings_initializer=random_normal(stddev=0.01), name='embedding')(embedding_inp)
    # -> (:, 1, 128)
    embedding = Reshape((1, 1, 128))(embedding)
    # -> (:, 1, 1, 128)

    # Decoder
    de_inp = concatenate([en_8, embedding], axis=3)
    # -> (:, 1, 1, 640)

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
    de_7 = concatenate([de_7, en_1], axis=3)
    # -> (:, 128, 128, 128)

    de_8 = Activation('relu')(de_7)
    de_8 = Conv2DTranspose(img_dim, (5, 5), strides=(2, 2), padding='same', kernel_initializer=random_normal(stddev=0.02))(de_8)
    de_8 = Activation('tanh')(de_8)
    # -> (:, 256, 256, img_dim)

    model = Model(inputs=[en_inp, embedding_inp], outputs=de_8)

    return model


def GeneratorDCGAN(img_size=(128, 128), img_dim=1, font_embedding_n=40, char_embedding_n=26, font_embedding_rate=0.5,
                   k_size=5, layer_n=3, smallest_hidden_unit_n=128, kernel_initializer=truncated_normal(), activation='relu',
                   is_bn=True):
    unit_size = img_size[0] // (2 ** layer_n)
    unit_n = smallest_hidden_unit_n * (2 ** (layer_n - 1))

    font_embedding_unit_n = int(unit_n * font_embedding_rate)
    char_embedding_unit_n = unit_n - font_embedding_unit_n

    font_embedding_inp = Input(shape=(1,), dtype='int32')
    font_embedding = Embedding(font_embedding_n, unit_size * unit_size * font_embedding_unit_n)(font_embedding_inp)
    font_embedding = Reshape((unit_size, unit_size, font_embedding_unit_n))(font_embedding)

    char_embedding_inp = Input(shape=(1,), dtype='int32')
    char_embedding = Embedding(char_embedding_n, unit_size * unit_size * char_embedding_unit_n)(char_embedding_inp)
    char_embedding = Reshape((unit_size, unit_size, char_embedding_unit_n))(char_embedding)

    x = concatenate([font_embedding, char_embedding], axis=3)

    for i in range(layer_n - 1):
        unit_n = smallest_hidden_unit_n * (2 ** (layer_n - i - 2))
        x = Conv2DTranspose(unit_n, (k_size, k_size), strides=(2, 2), padding='same', kernel_initializer=kernel_initializer)(x)
        if is_bn:
            x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = Activation(activation)(x)

    x = Conv2DTranspose(img_dim, (k_size, k_size), strides=(2, 2), padding='same', kernel_initializer=kernel_initializer)(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[font_embedding_inp, char_embedding_inp], outputs=x)

    return model


def DiscriminatorPix2Pix(img_size=(256, 256), img_dim=1, font_embedding_n=40):
    dis_inp = Input(shape=(img_size[0], img_size[1], img_dim))

    dis_1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(dis_inp)
    dis_1 = LeakyReLU(alpha=0.2)(dis_1)
    # -> (:, 128, 128, 64)

    dis_2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(dis_1)
    dis_2 = BatchNormalization(momentum=0.9, epsilon=0.00001)(dis_2)
    dis_2 = LeakyReLU(alpha=0.2)(dis_2)
    # -> (:, 64, 64, 128)

    dis_3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(dis_2)
    dis_3 = BatchNormalization(momentum=0.9, epsilon=0.00001)(dis_3)
    dis_3 = LeakyReLU(alpha=0.2)(dis_3)
    # -> (:, 32, 32, 256)

    dis_4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=truncated_normal(stddev=0.02))(dis_3)
    dis_4 = BatchNormalization(momentum=0.9, epsilon=0.00001)(dis_4)
    dis_4 = LeakyReLU(alpha=0.2)(dis_4)
    # -> (:, 16, 16, 512)

    fc_0 = Flatten(name='full_connected')(dis_4)
    # -> (:, 131072)

    model = Model(inputs=dis_inp, outputs=fc_0)

    return model


def DiscriminatorDCGAN(img_size=(128, 128), img_dim=1, k_size=5, layer_n=3, smallest_hidden_unit_n=128,
                       kernel_initializer=truncated_normal(), activation='leaky_relu', is_bn=True):
    dis_inp = Input(shape=(img_size[0], img_size[1], img_dim))
    x = Activation('linear')(dis_inp)

    for i in range(layer_n):
        unit_n = smallest_hidden_unit_n * (2 ** i)
        x = Conv2D(unit_n, (k_size, k_size), strides=(2, 2), padding='same', kernel_initializer=kernel_initializer)(x)
        if is_bn:
            x = BatchNormalization()(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(alpha=0.2)(x)
        else:
            x = Activation(activation)(x)

    x = Flatten()(x)

    x = Dense(1, activation=None)(x)

    model = Model(inputs=dis_inp, outputs=x)

    return model


def DiscriminatorBinarizeSubtract(discriminator, img_size=(256, 256), img_dim=1):
    real_inp = Input(shape=(img_size[0], img_size[1], img_dim))
    fake_inp = Input(shape=(img_size[0], img_size[1], img_dim))

    x = discriminator(real_inp)
    x = Dense(1, activation=None)(x)

    y = discriminator(fake_inp)
    y = Dense(1, activation=None)(y)

    z = Subtract()([x, y])

    model = Model(inputs=[real_inp, fake_inp], outputs=z)

    return model


def DiscriminatorBinarize(discriminator, img_size=(256, 256), img_dim=1):
    inp = Input(shape=(img_size[0], img_size[1], img_dim))

    x = discriminator(inp)
    x = Dense(1, activation=None)(x)

    model = Model(inputs=inp, outputs=x)

    return model


def DiscriminatorCategorize(discriminator, img_size=(256, 256), img_dim=1, font_embedding_n=40):
    inp = Input(shape=(img_size[0], img_size[1], img_dim))

    x = discriminator(inp)
    x = Dense(font_embedding_n, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)

    return model


def Classifier(img_size=(256, 256), img_dim=1, class_n=26):
    cl_inp = Input(shape=(img_size[0], img_size[0], img_dim))

    cl_1 = Conv2D(96, (8, 8), strides=(4, 4), padding='same')(cl_inp)
    cl_1 = Activation('relu')(cl_1)
    cl_1 = MaxPool2D((3, 3), strides=(2, 2))(cl_1)

    cl_2 = Conv2D(256, (5, 5), padding='same')(cl_1)
    cl_2 = Activation('relu')(cl_2)
    cl_2 = MaxPool2D((3, 3), strides=(2, 2))(cl_2)

    cl_3 = Conv2D(384, (3, 3), padding='same')(cl_2)
    cl_3 = Activation('relu')(cl_3)

    cl_4 = Conv2D(384, (3, 3), padding='same')(cl_3)
    cl_4 = Activation('relu')(cl_4)

    cl_5 = Conv2D(256, (3, 3), padding='same')(cl_4)
    cl_5 = Activation('relu')(cl_5)
    cl_5 = MaxPool2D((3, 3), strides=(2, 2))(cl_5)

    cl_6 = Flatten()(cl_5)
    cl_6 = Dense(4096)(cl_6)
    cl_6 = Dropout(0.5)(cl_6)

    cl_7 = Dense(4096)(cl_6)
    cl_7 = Dropout(0.5)(cl_7)

    cl_8 = Dense(class_n, activation='softmax')(cl_7)

    model = Model(inputs=cl_inp, outputs=cl_8)

    return model
