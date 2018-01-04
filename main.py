import os
import tensorflow as tf
from datetime import datetime
import subprocess

FLAGS = tf.app.flags.FLAGS


def get_gpu_n():
    result = subprocess.run('nvidia-smi -L | wc -l', shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        return 0
    return int(result.stdout)


def define_flags():
    now_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # Mode
    tf.app.flags.DEFINE_boolean('ttf2png', False, 'make dataset')
    tf.app.flags.DEFINE_boolean('png2dataset', False, 'make dataset')
    tf.app.flags.DEFINE_boolean('train_c', False, 'train classifier')
    tf.app.flags.DEFINE_boolean('test_c', False, 'test with classifier')
    tf.app.flags.DEFINE_boolean('train_g', False, 'train GAN')
    tf.app.flags.DEFINE_boolean('generate', False, 'generate images')
    tf.app.flags.DEFINE_boolean('generate_test', False, 'for recognition test')
    tf.app.flags.DEFINE_boolean('intermediate', False, 'visualize intermediate layers')
    tf.app.flags.DEFINE_boolean('evaluate', False, 'evaluate fonts')

    # Common
    tf.app.flags.DEFINE_string('gpu_ids', ', '.join([str(i) for i in range(get_gpu_n())]), 'using GPU ids')
    tf.app.flags.DEFINE_string('font_h5', '', 'path of real fonts hdf5')
    tf.app.flags.DEFINE_integer('img_width', 64, 'image\'s width')
    tf.app.flags.DEFINE_integer('img_height', 64, 'image\'\'s height')
    tf.app.flags.DEFINE_integer('img_dim', 3, 'image\'s dimention')
    tf.app.flags.DEFINE_integer('font_embedding_n', 256, 'num of font embedding ids')
    tf.app.flags.DEFINE_string('embedding_chars_type', 'caps', 'embedding type of characters')
    tf.app.flags.DEFINE_integer('font_z_size', 100, 'z size')
    tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')

    # Common Directories
    gan_dir = 'result/gan/' + now_str
    classifier_dir = 'result/classifier/' + now_str
    font_pngs_dir = 'src/pngs/' + now_str
    tf.app.flags.DEFINE_string('gan_dir', gan_dir, 'path of result\'s destination')
    tf.app.flags.DEFINE_string('classifier_dir', classifier_dir, 'path of result\'s destination')
    tf.app.flags.DEFINE_string('font_pngs', font_pngs_dir, 'path of font images\' directory')

    # ttf to png
    tf.app.flags.DEFINE_string('font_ttfs', '', 'path of font files\' directory')

    # png to dataset

    # Train Classifier
    tf.app.flags.DEFINE_float('train_rate', 0.9, 'train:test = train_rate:(1. - train_rate)')
    tf.app.flags.DEFINE_integer('c_epoch_n', 10, 'num of epoch for training classifier')
    tf.app.flags.DEFINE_boolean('labelacc', False, 'accuracy by labels')

    # Train GAN
    tf.app.flags.DEFINE_integer('gan_epoch_n', 10000, 'num of epoch for training GAN')
    tf.app.flags.DEFINE_integer('critic_n', 5, 'num of critics to approximate wasserstein distance')
    tf.app.flags.DEFINE_integer('sample_imgs_interval', 10, 'interval epochs of saving images')
    tf.app.flags.DEFINE_integer('sample_col_n', 26, 'sample images\' column num')
    tf.app.flags.DEFINE_integer('ckpt_keep_n', 5, 'num of keeping ckpts')
    tf.app.flags.DEFINE_integer('keep_ckpt_hour', 12, 'hour of keeping ckpts')
    tf.app.flags.DEFINE_integer('keep_ckpt_interval', 250, 'interval of keeping ckpts')
    tf.app.flags.DEFINE_string('arch', 'DCGAN', 'archtect of GAN')
    tf.app.flags.DEFINE_boolean('transpose', False, 'use conv2d_transpose or resize_bilinear')
    tf.app.flags.DEFINE_boolean('batchnorm', False, 'use batchnorm in Generator and Discriminator or not')
    tf.app.flags.DEFINE_boolean('run_tensorboard', True, 'run tensorboard or not')
    tf.app.flags.DEFINE_integer('tensorboard_port', 6006, 'port of tensorboard')

    # Generate GAN
    tf.app.flags.DEFINE_string('src_ids', '', 'path of ids settings\' json')
    tf.app.flags.DEFINE_string('gen_name', now_str, 'filename of saveing image')
    tf.app.flags.DEFINE_integer('char_img_n', 256, 'one chars\' img num for recognition test')

    # Intermediate
    tf.app.flags.DEFINE_string('plot_method', 'TSNE', 'TSNE or MDS')
    tf.app.flags.DEFINE_integer('tsne_p', 30, 'TNSE\'s perplexity')

    # Evaluate
    tf.app.flags.DEFINE_string('generated_h5', '', 'path of generated fonts hdf5')


def main(argv=None):
    if FLAGS.ttf2png:
        assert FLAGS.font_ttfs != '', 'have to set --font_ttfs'
        from font2img.font2img import font2img
        if 'hiragana' in FLAGS.embedding_chars_type:
            src_chars_txt_path = 'font2img/src_chars_txt/hiragana_seion.txt'
        else:
            src_chars_txt_path = 'font2img/src_chars_txt/alphabets_hankaku_caps.txt'
        if not os.path.exists(FLAGS.font_pngs):
            os.makedirs(FLAGS.font_pngs)
        f2i = font2img(src_font_dir_path=FLAGS.font_ttfs,
                       src_chars_txt_path=src_chars_txt_path,
                       dst_dir_path=FLAGS.font_pngs,
                       canvas_size=FLAGS.img_height,
                       font_size=0,
                       output_ext='png',
                       is_center=True,
                       is_maximum=False,
                       is_binary=False,
                       is_unicode=False,
                       is_by_char=True,
                       is_recursive=True)
        f2i.run()
    if FLAGS.png2dataset:
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        from dataset import Dataset
        dataset = Dataset(FLAGS.font_h5, 'w', FLAGS.img_width, FLAGS.img_height, FLAGS.img_dim)
        dataset.load_imgs(FLAGS.font_pngs)
        del dataset
    if FLAGS.train_c:
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        from train_classifier import TrainingClassifier
        cl = TrainingClassifier()
        cl.train_and_test()
        del cl
    if FLAGS.test_c:
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        from train_classifier import TrainingClassifier
        cl = TrainingClassifier()
        cl.test()
        del cl
    if FLAGS.train_g:
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        from train_gan import TrainingFontDesignGAN
        gan = TrainingFontDesignGAN()
        gan.train()
        del gan
    if FLAGS.generate:
        assert FLAGS.gan_dir != '', 'have to set --gan_dir'
        assert FLAGS.src_ids != '', 'have to set --src_ids'
        from generate import GeneratingFontDesignGAN
        gan = GeneratingFontDesignGAN()
        gan.generate(filename=FLAGS.gen_name)
        del gan
    if FLAGS.generate_test:
        assert FLAGS.gan_dir != '', 'have to set --gan_dir'
        from generate import GeneratingFontDesignGAN
        gan = GeneratingFontDesignGAN()
        gan.generate_for_recognition_test()
        del gan
    if FLAGS.intermediate:
        assert FLAGS.gan_dir != '', 'have to set --gan_dir'
        assert FLAGS.src_ids != '', 'have to set --src_ids'
        from generate import GeneratingFontDesignGAN
        gan = GeneratingFontDesignGAN()
        gan.visualize_intermediate(FLAGS.gen_name)
        del gan
    if FLAGS.evaluate:
        assert FLAGS.font_h5 != '', 'have to set --font_h5'
        assert FLAGS.generated_h5 != '', 'have to set --generated_h5'
        from evaluate import Evaluating
        evaluating = Evaluating()
        evaluating.calc_hamming_distance()
        del evaluating


if __name__ == '__main__':
    define_flags()
    tf.app.run()
