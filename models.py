from keras.models import Model
from keras.layers import Input, Activation, Dropout, Dense, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

# def discriminator_model():
#     model = Sequential()
#     model.add(Convolution2D(
#                         64, 4, 4,
#                         border_mode='same',
#                         input_shape=(1, 28, 28)))
#     model.add(Activation('tanh'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Convolution2D(128, 4, 4))
#     model.add(Activation('tanh'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(1024))
#     model.add(Activation('tanh'))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     return model

class Encoder(Model):
    def __init__(self, src_shape):
        self._build(src_shape):
        super(Encoder, self).__init__(inputs=self.inp, outputs=self.layer8)

    def _build(self):
        self.inp = Input(shape=src_shape)

        self.layer1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inp)

        self.layer2 = LeakyReLU(alpha=0.2)(self.layer1)
        self.layer2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(self.layer2)
        self.layer2 = BatchNormalization()(self.layer2)

        self.layer3 = LeakyReLU(alpha=0.2)(self.layer2)
        self.layer3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(self.layer3)
        self.layer3 = BatchNormalization()(self.layer2)

        self.layer4 = LeakyReLU(alpha=0.2)(self.layer3)
        self.layer4 = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(self.layer4)
        self.layer4 = BatchNormalization()(self.layer4)

        self.layer5 = LeakyReLU(alpha=0.2)(self.layer4)
        self.layer5 = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(self.layer5)
        self.layer5 = BatchNormalization()(self.layer5)

        self.layer6 = LeakyReLU(alpha=0.2)(self.layer5)
        self.layer6 = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(self.layer6)
        self.layer6 = BatchNormalization()(self.layer6)

        self.layer7 = LeakyReLU(alpha=0.2)(self.layer6)
        self.layer7 = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(self.layer7)
        self.layer7 = BatchNormalization()(self.layer7)

        self.layer8 = LeakyReLU(alpha=0.2)(self.layer7)
        self.layer8 = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(self.layer8)
        self.layer8 = BatchNormalization()(self.layer8)

class Decoder(Sequential):
    def __init__(self, encoder, gen_shpae):
        self._build(encoder, gen_shape):
        super(Decoder, self).__init__(inputs=self.inp, outputs=self.layer8)

    def _build(self, encoder, gen_shape):
        self.inp = Input(shape=gen_shape)

        self.layer1 = Activation('relu')(self.inp)
        self.layer1 = Conv2DTranspose(512, (4, 4), strides=(2, 2))(self.layer1)
        self.layer1 = BatchNormalization()(self.layer1)
        self.layer1 = Dropout(0.5)(self.layer1)
        self.layer1 = concatenate([self.layer1, encoder.layer7], axis=1)

        self.layer2 = Activation('relu')(self.layer1)
        self.layer2 = Conv2DTranspose(1024, (4, 4), strides=(2, 2))(self.layer2)
        self.layer2 = BatchNormalization()(self.layer2)
        self.layer2 = Dropout(0.5)(self.layer2)
        self.layer2 = concatenate([self.layer2, encoder.layer6], axis=1)

        self.layer3 = Activation('relu')(self.layer2)
        self.layer3 = Conv2DTranspose(1024, (4, 4), strides=(2, 2))(self.layer3)
        self.layer3 = BatchNormalization()(self.layer3)
        self.layer3 = Dropout(0.5)(self.layer3)
        self.layer3 = concatenate([self.layer3, encoder.layer5], axis=1)

        self.layer4 = Activation('relu')(self.layer3)
        self.layer4 = Conv2DTranspose(1024, (4, 4), strides=(2, 2))(self.layer4)
        self.layer4 = BatchNormalization()(self.layer4)
        self.layer4 = concatenate([self.layer4, encoder.layer4], axis=1)

        self.layer5 = Activation('relu')(self.layer4)
        self.layer5 = Conv2DTranspose(1024, (4, 4), strides=(2, 2))(self.layer5)
        self.layer5 = BatchNormalization()(self.layer5)
        self.layer5 = concatenate([self.layer5, encoder.layer3], axis=1)

        self.layer6 = Activation('relu')(self.layer5)
        self.layer6 = Conv2DTranspose(512, (4, 4), strides=(2, 2))(self.layer6)
        self.layer6 = BatchNormalization()(self.layer6)
        self.layer6 = concatenate([self.layer6, encoder.layer2], axis=1)

        self.layer7 = Activation('relu')(self.layer7)
        self.layer7 = Conv2DTranspose(256, (4, 4), strides=(2, 2))(self.layer7)
        self.layer7 = BatchNormalization()(self.layer7)
        self.layer7 = concatenate([self.layer7, encoder.layer1], axis=1)

        self.layer8 = Activation('relu')(self.layer7)
        self.layer8 = Conv2DTranspose(1, (4, 4), strides=(2, 2))(self.layer8)

