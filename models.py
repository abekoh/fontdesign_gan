from keras.models import Model
from keras.layers import Input, Activation, Dropout, Embedding, Reshape, Flatten, Dense, concatenate, MaxPool2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.initializers import truncated_normal
from ops import sign


def GeneratorDCGAN(img_size=(128, 128), img_dim=1, z_size=100,
                   k_size=5, layer_n=3, smallest_hidden_unit_n=128, kernel_initializer=truncated_normal(), activation='relu',
                   output_activation='tanh', is_bn=True):
    unit_size = img_size[0] // (2 ** layer_n)
    unit_n = smallest_hidden_unit_n * (2 ** (layer_n - 1))

    z_inp = Input(shape=(z_size,))
    x = Dense(unit_size * unit_size * unit_n)(z_inp)
    if is_bn:
        x = BatchNormalization()(x)
    if activation == 'leaky_relu':
        x = LeakyReLU(alpha=0.2)(x)
    else:
        x = Activation(activation)(x)
    x = Reshape((unit_size, unit_size, unit_n))(x)

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
    if output_activation == 'sign':
        x = Activation(sign)(x)
    else:
        x = Activation(output_activation)(x)

    model = Model(inputs=z_inp, outputs=x)

    return model


def GeneratorDCGANWithEmbedding(img_size=(128, 128), img_dim=1, font_embedding_n=40, char_embedding_n=26, font_embedding_rate=0.5,
                                k_size=5, layer_n=3, smallest_hidden_unit_n=128, kernel_initializer=truncated_normal(), activation='relu',
                                output_activation='tanh', is_bn=True):
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
    if output_activation == 'sign':
        x = Activation(sign)(x)
    else:
        x = Activation(output_activation)(x)

    model = Model(inputs=[font_embedding_inp, char_embedding_inp], outputs=x)

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


def ClassifierMin(img_size=(64, 64), img_dim=1, class_n=26):
    inp = Input(shape=(img_size[0], img_size[0], img_dim))

    x = Conv2D(96, (5, 5), strides=(2, 2), padding='same')(inp)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(256, (5, 5), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    x = Conv2D(384, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(4096)(x)
    x = Dropout(0.5)(x)

    x = Dense(class_n, activation='softmax')(x)

    model = Model(inputs=inp, outputs=x)

    return model
