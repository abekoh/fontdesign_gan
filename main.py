import tensorflow as tf
from datetime import datetime

from train_classifier import TrainingClassifier
from train_gan import TrainingFontDesignGAN
from generate import GeneratingFontDesignGAN

FLAGS = tf.app.flags.FLAGS


def define_flags():

    # Mode
    tf.app.flags.DEFINE_string('mode', '', 'train_c or train_g or generate')
    tf.app.flags.DEFINE_string('gpu_ids', '0', 'gpu ids')
    tf.app.flags.DEFINE_integer('gpu_n', 1, 'gpu num')

    # Directory Path
    now_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    dst_gan = 'result/gan/' + now_str
    dst_classifier = 'result/classifier/' + now_str

    # Source Paths
    tf.app.flags.DEFINE_string('font_h5', 'src/fonts_6627_caps_3ch_64x64.h5', 'source path of real fonts hdf5')
    tf.app.flags.DEFINE_string('classifier_ckpt', 'result_classifier/current/log/result_9.ckpt', 'source path of classifier ckpt')
    tf.app.flags.DEFINE_string('gan_ckpt', 'result_pickup/2017-10-22_032515/log/result_7.ckpt', 'source path of result ckpt')
    # Destination Paths
    tf.app.flags.DEFINE_string('dst_gan', dst_gan, 'destination path')
    tf.app.flags.DEFINE_string('dst_classifier', dst_classifier, 'destination classifier-mode path')

    # Images Settings
    tf.app.flags.DEFINE_integer('img_width', 64, 'img width')
    tf.app.flags.DEFINE_integer('img_height', 64, 'img height')
    tf.app.flags.DEFINE_integer('img_dim', 3, 'img dimention')
    tf.app.flags.DEFINE_integer('font_embedding_n', 256, 'font embedding num')
    tf.app.flags.DEFINE_integer('char_embedding_n', 26, 'char embeddings num')
    tf.app.flags.DEFINE_float('font_embedding_rate', 0.5, 'font embedding rate')

    # Networks Settings
    tf.app.flags.DEFINE_float('c_penalty', 0.01, 'learning penalty of classifier')
    tf.app.flags.DEFINE_float('c_lr', 0.0000025, 'learning rate of generator iwth classifier')

    # Train Classifier Settings
    tf.app.flags.DEFINE_integer('train_rate', 0.9, 'train:test = train_rate:(1. - train_rate)')

    # Train GAN Settings
    tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')
    tf.app.flags.DEFINE_integer('epoch_n', 150000, 'epoch cycles')
    tf.app.flags.DEFINE_integer('critic_n', 5, 'how many critic wasserstein distance')
    tf.app.flags.DEFINE_integer('z_size', 100, 'z size')

    # Verbose
    tf.app.flags.DEFINE_integer('sample_imgs_interval', 20, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('embedding_interval', 20, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('ckpt_keep_n', 5, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('keep_ckpt_hour', 4, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('save_imgs_col_n', 16, 'column num of save imgs')
    tf.app.flags.DEFINE_boolean('is_run_tensorboard', True, 'run tensorboard or not')


def main(argv=None):
    if FLAGS.mode == 'train_c':
        cl = TrainingClassifier()
        cl.setup()
        cl.train()
    if FLAGS.mode == 'train_g':
        gan = TrainingFontDesignGAN()
        gan.setup()
        gan.train()
    if FLAGS.mode == 'generate':
        gan = GeneratingFontDesignGAN('./gen_sample.json')
        gan.setup()
        gan.generate(FLAGS.gen_filename)


if __name__ == '__main__':
    define_flags()
    tf.app.run()
