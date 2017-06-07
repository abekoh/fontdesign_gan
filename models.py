from keras.models import Model
from keras.layers import Input, Activation, Dropout, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization


def Encoder():

    inp = Input(shape=(256, 256, 3))

    layer1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='en_l1')(inp)
    # -> (:, 128, 128, 64)

    layer2 = LeakyReLU(alpha=0.2)(layer1)
    layer2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(layer2)
    layer2 = BatchNormalization(name='en_l2')(layer2)
    # -> (:, 64, 64, 128)

    layer3 = LeakyReLU(alpha=0.2)(layer2)
    layer3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(layer3)
    layer3 = BatchNormalization(name='en_l3')(layer3)
    # -> (:, 32, 32, 256)

    layer4 = LeakyReLU(alpha=0.2)(layer3)
    layer4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(layer4)
    layer4 = BatchNormalization(name='en_l4')(layer4)
    # -> (:, 16, 16, 512)

    layer5 = LeakyReLU(alpha=0.2)(layer4)
    layer5 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(layer5)
    layer5 = BatchNormalization(name='en_l5')(layer5)
    # -> (:, 8, 8, 512)

    layer6 = LeakyReLU(alpha=0.2)(layer5)
    layer6 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(layer6)
    layer6 = BatchNormalization(name='en_l6')(layer6)
    # -> (:, 4, 4, 512)

    layer7 = LeakyReLU(alpha=0.2)(layer6)
    layer7 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(layer7)
    layer7 = BatchNormalization(name='en_l7')(layer7)
    # -> (:, 2, 2, 512)

    layer8 = LeakyReLU(alpha=0.2)(layer7)
    layer8 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(layer8)
    layer8 = BatchNormalization(name='en_l8')(layer8)
    # -> (:, 1, 1, 512)

    model = Model(inputs=inp, outputs=layer8)

    return model


def Decoder(encoder):

    inp = Input(shape=(1, 1, 128))

    layer1 = Activation('relu')(inp)
    layer1 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(layer1)
    layer1 = BatchNormalization()(layer1)
    layer1 = Dropout(0.5)(layer1)
    layer1 = concatenate([layer1, encoder.get_layer('en_l7').output], axis=3)

    layer2 = Activation('relu')(layer1)
    layer2 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(layer2)
    layer2 = BatchNormalization()(layer2)
    layer2 = Dropout(0.5)(layer2)
    layer2 = concatenate([layer2, encoder.get_layer('en_l6').output], axis=3)

    layer3 = Activation('relu')(layer2)
    layer3 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(layer3)
    layer3 = BatchNormalization()(layer3)
    layer3 = Dropout(0.5)(layer3)
    layer3 = concatenate([layer3, encoder.get_layer('en_l5').output], axis=3)

    layer4 = Activation('relu')(layer3)
    layer4 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(layer4)
    layer4 = BatchNormalization()(layer4)
    layer4 = concatenate([layer4, encoder.get_layer('en_l4').output], axis=3)

    layer5 = Activation('relu')(layer4)
    layer5 = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(layer5)
    layer5 = BatchNormalization()(layer5)
    layer5 = concatenate([layer5, encoder.get_layer('en_l3').output], axis=3)

    layer6 = Activation('relu')(layer5)
    layer6 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(layer6)
    layer6 = BatchNormalization()(layer6)
    layer6 = concatenate([layer6, encoder.get_layer('en_l2').output], axis=3)

    layer7 = Activation('relu')(layer6)
    layer7 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(layer7)
    layer7 = BatchNormalization()(layer7)
    layer7 = concatenate([layer7, encoder.get_layer('en_l1').output], axis=3)

    layer8 = Activation('relu')(layer7)
    layer8 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(layer8)

    model = Model(inputs=inp, outputs=layer8)

    return model
