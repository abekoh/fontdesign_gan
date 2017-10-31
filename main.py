import tensorflow as tf
from datetime import datetime

from dataset import Dataset
from train_classifier import TrainingClassifier
from train_gan import TrainingFontDesignGAN
from generate import GeneratingFontDesignGAN

FLAGS = tf.app.flags.FLAGS


def define_flags():
    now_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # Mode
    tf.app.flags.DEFINE_string('mode', '', 'train_c or train_g or generate')

    # Common
    tf.app.flags.DEFINE_string('gpu_ids', '0', 'gpu ids')
    tf.app.flags.DEFINE_integer('gpu_n', 1, 'gpu num')
    tf.app.flags.DEFINE_string('font_h5', 'src/fonts_6627_caps_3ch_64x64.h5', 'source path of real fonts hdf5')
    tf.app.flags.DEFINE_integer('img_width', 64, 'img width')
    tf.app.flags.DEFINE_integer('img_height', 64, 'img height')
    tf.app.flags.DEFINE_integer('img_dim', 3, 'img dimention')
    tf.app.flags.DEFINE_integer('font_embedding_n', 256, 'font embedding num')
    tf.app.flags.DEFINE_integer('char_embedding_n', 26, 'char embeddings num')
    tf.app.flags.DEFINE_float('font_embedding_rate', 0.5, 'font embedding rate')
    tf.app.flags.DEFINE_integer('z_size', 100, 'z size')
    tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')

    # Dataset
    tf.app.flags.DEFINE_string('src_font_imgs', '../../font_dataset/png/6628_64x64', 'source path of result ckpt')

    # Train Classifier
    dst_classifier = 'result/classifier/' + now_str
    tf.app.flags.DEFINE_string('dst_classifier', dst_classifier, 'destination classifier-mode path')
    tf.app.flags.DEFINE_integer('train_rate', 0.9, 'train:test = train_rate:(1. - train_rate)')
    tf.app.flags.DEFINE_integer('c_epoch_n', 10, 'epoch cycles')

    # Train GAN
    dst_gan = 'result/gan/' + now_str
    tf.app.flags.DEFINE_string('dst_gan', dst_gan, 'destination path')
    tf.app.flags.DEFINE_string('src_classifier', 'result_classifier/current', 'source path of classifier ckpt')
    tf.app.flags.DEFINE_float('c_penalty', 0.01, 'learning penalty of classifier')
    tf.app.flags.DEFINE_float('c_lr', 0.0000025, 'learning rate of generator iwth classifier')
    tf.app.flags.DEFINE_integer('gan_epoch_n', 150000, 'epoch cycles')
    tf.app.flags.DEFINE_integer('critic_n', 5, 'how many critic wasserstein distance')
    tf.app.flags.DEFINE_integer('sample_imgs_interval', 20, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('embedding_interval', 20, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('ckpt_keep_n', 5, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('keep_ckpt_hour', 4, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('save_imgs_col_n', 16, 'column num of save imgs')
    tf.app.flags.DEFINE_boolean('is_run_tensorboard', True, 'run tensorboard or not')

    # Generate GAN
    tf.app.flags.DEFINE_string('src_gan', 'result_pickup/2017-10-26_070102', 'source path of result ckpt')
    tf.app.flags.DEFINE_string('gen_name', now_str + '.png', 'destination classifier-mode path')


def main(argv=None):
    if FLAGS.mode == 'make_dataset':
        dataset = Dataset(FLAGS.font_h5, 'w', img_size=(FLAGS.img_width, FLAGS.img_height), img_dim=FLAGS.img_dim)
        dataset.load_imgs(FLAGS.src_font_imgs)
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
        gan.generate(FLAGS.gen_name)


if __name__ == '__main__':
    define_flags()
    tf.app.run()
