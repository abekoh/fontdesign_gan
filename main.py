import tensorflow as tf
from datetime import datetime
import subprocess

FLAGS = tf.app.flags.FLAGS
ALPHABET_CAPS = tuple(chr(i) for i in range(65, 65 + 26))


def get_gpu_n():
    result = subprocess.run('nvidia-smi -L | wc -l', shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        return 0
    return int(result.stdout)


def define_flags():
    now_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # Mode
    tf.app.flags.DEFINE_string('mode', '', 'make_dataset or train_c or train_g or generate')

    # Common
    tf.app.flags.DEFINE_string('gpu_ids', ', '.join([str(i) for i in range(get_gpu_n())]), 'using GPU ids')
    tf.app.flags.DEFINE_string('font_h5', '', 'path of real fonts hdf5')
    tf.app.flags.DEFINE_integer('img_width', 64, 'image\'s width')
    tf.app.flags.DEFINE_integer('img_height', 64, 'image\'\'s height')
    tf.app.flags.DEFINE_integer('img_dim', 3, 'image\'s dimention')
    tf.app.flags.DEFINE_integer('font_embedding_n', 256, 'num of font embedding ids')
    tf.app.flags.DEFINE_integer('embedding_chars', ALPHABET_CAPS, 'char embeddings')
    tf.app.flags.DEFINE_float('font_embedding_rate', 0.5, 'rate of font embedding')
    tf.app.flags.DEFINE_integer('z_size', 100, 'z size')
    tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')

    # Make Dataset
    tf.app.flags.DEFINE_string('font_imgs', '', 'path of font images\' directory')

    # Train Classifier
    dst_classifier = 'result/classifier/' + now_str
    tf.app.flags.DEFINE_string('dst_classifier', dst_classifier, 'path of result\'s destination')
    tf.app.flags.DEFINE_integer('train_rate', 0.9, 'train:test = train_rate:(1. - train_rate)')
    tf.app.flags.DEFINE_integer('c_epoch_n', 10, 'num of epoch for training classifier')

    # Train GAN
    dst_gan = 'result/gan/' + now_str
    tf.app.flags.DEFINE_string('dst_gan', dst_gan, 'path of result\'s destination')
    tf.app.flags.DEFINE_string('src_classifier', '', 'path of trained classifier\'s result directory')
    tf.app.flags.DEFINE_float('c_penalty', 0.01, 'training penalty of classifier')
    tf.app.flags.DEFINE_float('c_lr', 0.00001, 'training rate of generator with classifier')
    tf.app.flags.DEFINE_integer('gan_epoch_n', 10000, 'num of epoch for training GAN')
    tf.app.flags.DEFINE_integer('critic_n', 5, 'num of critics to approximate wasserstein distance')
    tf.app.flags.DEFINE_integer('sample_imgs_interval', 1, 'interval epochs of saving images')
    tf.app.flags.DEFINE_integer('embedding_interval', 20, 'interval epochs of saving embedding images')
    tf.app.flags.DEFINE_integer('ckpt_keep_n', 5, 'num of keeping ckpts')
    tf.app.flags.DEFINE_integer('keep_ckpt_hour', 4, 'hours of keeping ckpts')
    tf.app.flags.DEFINE_boolean('transpose', False, 'use conv2d_transpose or resize_bilinear')
    tf.app.flags.DEFINE_boolean('batchnorm', False, 'use batchnorm in Generator and Discriminator or not')
    tf.app.flags.DEFINE_boolean('run_tensorboard', True, 'run tensorboard or not')
    tf.app.flags.DEFINE_integer('tensorboard_port', 6006, 'port of tensorboard')

    # Generate GAN
    tf.app.flags.DEFINE_string('src_gan', '', 'path of trained gan\'s result directory')
    tf.app.flags.DEFINE_string('src_ids', '', 'path of ids settings\' json')
    tf.app.flags.DEFINE_string('gen_name', now_str + '.png', 'filename of saveing image')


def main(argv=None):
    if FLAGS.mode == 'make_dataset':
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        assert FLAGS.font_imgs != '', 'have to set --font_imgs'
        from dataset import Dataset
        dataset = Dataset(FLAGS.font_h5, 'w', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        dataset.load_imgs(FLAGS.font_imgs)
    elif FLAGS.mode == 'train_c':
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        from train_classifier import TrainingClassifier
        cl = TrainingClassifier()
        cl.train()
    elif FLAGS.mode == 'train_g':
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        if FLAGS.c_penalty != 0.:
            assert FLAGS.src_classifier != '', 'have to set --src_classifier'
        from train_gan import TrainingFontDesignGAN
        gan = TrainingFontDesignGAN()
        gan.train()
    elif FLAGS.mode == 'generate':
        assert FLAGS.src_gan != '', 'have to set --src_gan'
        assert FLAGS.src_ids != '', 'have to set --src_ids'
        from generate import GeneratingFontDesignGAN
        gan = GeneratingFontDesignGAN()
        gan.generate(FLAGS.gen_name)
    else:
        print('set --mode {make_dataset train_c train_g generate}')


if __name__ == '__main__':
    define_flags()
    tf.app.run()
