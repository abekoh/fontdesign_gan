from keras.models import Model
from models import Generator, Discriminator
from keras.optimizers import Adam


def train():
    discriminator = Discriminator()
    discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                          loss=['binary_crossentropy', 'categorical_crossentropy'],
                          loss_weights=[1., 0.5])

    generator = Generator()
    generator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                      loss='mean_absolute_error', loss_weights=100.)

    discriminator.trainable = False
    generator_to_discriminator = Model(inputs=generator.input, outputs=discriminator(generator.output))
    generator_to_discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                                       loss=['binary_crossentropy', 'categorical_crossentropy'],
                                       loss_weights=[1., 1.])

    encoder = Model(inputs=generator.input, outputs=generator.get_layer('en_8').output)
    encoder.trainable = False
    generator_to_encoder = Model(inputs=generator.input, outputs=encoder(generator.output))
    generator_to_encoder.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                                 loss='mean_squared_error',
                                 loss_weight=[1.])

    for epoch_i in range(epoch_n):
