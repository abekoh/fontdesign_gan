from keras.models import Model
from keras.layers import Input, Activation, Dropout, concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

# def discriminator_model():
#     model = Sequential()
#     model.add(Convolution2D(
#                         65, 5, 4,
#                         border_mode='same',
#                         input_shape=(1, 28, 28)))
#     model.add(Activation('tanh'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Convolution2D(128, 5, 5))
#     model.add(Activation('tanh'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(1024))
#     model.add(Activation('tanh'))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     return model


class Encoder(Model):

    def __init__(self, output_name='en_l8'):
        self._build()
        super(Encoder, self).__init__(inputs=self.inp, outputs=Model.get_layer(self, name=output_name).outpu)

    def _build(self):
        self.inp = Input(shape=(256, 256, 3))

        self.layer1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='en_l1')(self.inp)
        # -> (None, 256, 256, 64)

        self.layer2 = LeakyReLU(alpha=0.2)(self.layer1)
        self.layer2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(self.layer2)
        self.layer2 = BatchNormalization(name='en_l2')(self.layer2)
        # -> (None, 64, 64, 128)

        self.layer3 = LeakyReLU(alpha=0.2)(self.layer2)
        self.layer3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(self.layer3)
        self.layer3 = BatchNormalization(name='en_l3')(self.layer2)
        # -> (None, 32, 32, 256)

        self.layer4 = LeakyReLU(alpha=0.2)(self.layer3)
        self.layer4 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(self.layer4)
        self.layer4 = BatchNormalization(name='en_l4')(self.layer4)
        # -> (None, 16, 16, 512)

        self.layer5 = LeakyReLU(alpha=0.2)(self.layer4)
        self.layer5 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(self.layer5)
        self.layer5 = BatchNormalization(name='en_l5')(self.layer5)
        # -> (None, 8, 8, 512)

        self.layer6 = LeakyReLU(alpha=0.2)(self.layer5)
        self.layer6 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(self.layer6)
        self.layer6 = BatchNormalization(name='en_l6')(self.layer6)
        # -> (None, 4, 4, 512)

        self.layer7 = LeakyReLU(alpha=0.2)(self.layer6)
        self.layer7 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(self.layer7)
        self.layer7 = BatchNormalization(name='en_l7')(self.layer7)
        # -> (None, 2, 2, 512)

        self.layer8 = LeakyReLU(alpha=0.2)(self.layer7)
        self.layer8 = Conv2D(512, (5, 5), strides=(2, 2), padding='same')(self.layer8)
        self.layer8 = BatchNormalization(name='en_l8')(self.layer8)
        # -> (None, 1, 1, 512)


class Decoder(Model):

    def __init__(self, encoder):
        self._build(encoder)
        super(Decoder, self).__init__(inputs=self.inp, outputs=self.layer8)

    def _build(self, encoder):
        self.inp = Input(shape=(1, 1, 128))

        self.layer1 = Activation('relu')(self.inp)
        self.layer1 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(self.layer1)
        self.layer1 = BatchNormalization()(self.layer1)
        self.layer1 = Dropout(0.5)(self.layer1)
        self.layer1 = concatenate([self.layer1, encoder.layer8], axis=3)

        self.layer2 = Activation('relu')(self.layer1)
        self.layer2 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(self.layer2)
        self.layer2 = BatchNormalization()(self.layer2)
        self.layer2 = Dropout(0.5)(self.layer2)
        self.layer2 = concatenate([self.layer2, encoder.layer7], axis=3)

        self.layer3 = Activation('relu')(self.layer2)
        self.layer3 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(self.layer3)
        self.layer3 = BatchNormalization()(self.layer3)
        self.layer3 = Dropout(0.5)(self.layer3)
        self.layer3 = concatenate([self.layer3, encoder.layer6], axis=3)

        self.layer4 = Activation('relu')(self.layer3)
        self.layer4 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(self.layer4)
        self.layer4 = BatchNormalization()(self.layer4)
        self.layer4 = Dropout(0.5)(self.layer4)
        self.layer4 = concatenate([self.layer4, encoder.layer5], axis=3)

        self.layer5 = Activation('relu')(self.layer4)
        self.layer5 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same')(self.layer5)
        self.layer5 = BatchNormalization()(self.layer5)
        self.layer5 = concatenate([self.layer5, encoder.layer4], axis=3)

        self.layer6 = Activation('relu')(self.layer5)
        self.layer6 = Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same')(self.layer6)
        self.layer6 = BatchNormalization()(self.layer6)
        self.layer6 = concatenate([self.layer6, encoder.layer3], axis=3)

        self.layer7 = Activation('relu')(self.layer6)
        self.layer7 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(self.layer7)
        self.layer7 = BatchNormalization()(self.layer7)
        self.layer7 = concatenate([self.layer7, encoder.layer2], axis=3)

        self.layer8 = Activation('relu')(self.layer7)
        self.layer8 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same')(self.layer8)
        self.layer8 = BatchNormalization()(self.layer8)
        self.layer8 = concatenate([self.layer8, encoder.layer1], axis=3)

        self.layer9 = Activation('relu')(self.layer7)
        self.layer9 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(self.layer9)

# class Generator(Model):
#     def __init__(self, encoder):
#         self._build(encoder)
#         super(Generator, self).__init__(inputs=[self.inp, self.fontid], outputs=self.embed)
#
#     def _build(self, encoder):
#         self.inp = Input(shape=(256, 256, 1))
#         self.encoded = encoder(self.inp)
#         self.fontid = Input(shape=(1))
#         self.emb = Embedding(12sh, 512, input_length=40)(self.fontid)
#         self.embed = concatenate([self.encoded, self.emb])
