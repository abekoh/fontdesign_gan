import tensorflow as tf
from datetime import datetime

from train_gan import TrainingFontDesignGAN
from generate import GeneratingFontDesignGAN

FLAGS = tf.app.flags.FLAGS


def define_flags():
    # Directory Path
    dst_root = 'result/' + datetime.now().strftime('%Y-%m-%d_%H%M%S')

    # Mode
    tf.app.flags.DEFINE_string('mode', 'train', 'train or generate')
    # Images Settings
    tf.app.flags.DEFINE_integer('img_width', 64, 'img width')
    tf.app.flags.DEFINE_integer('img_height', 64, 'img height')
    tf.app.flags.DEFINE_integer('img_dim', 3, 'img dimention')
    tf.app.flags.DEFINE_integer('font_embedding_n', 256, 'font embedding num')
    tf.app.flags.DEFINE_integer('char_embedding_n', 26, 'char embeddings num')
    tf.app.flags.DEFINE_float('font_embedding_rate', 0.5, 'font embedding rate')
    # Networks Settings
    # - Generator
    tf.app.flags.DEFINE_integer('g_layer_n', 4, 'layer num of generator')
    tf.app.flags.DEFINE_integer('g_smallest_hidden_unit_n', 64, 'smallest hidden unit num of generator')
    tf.app.flags.DEFINE_integer('g_k_size', 3, 'kernel size of generator')
    # - Discriminator
    tf.app.flags.DEFINE_integer('d_layer_n', 4, 'layer num of discriminator')
    tf.app.flags.DEFINE_integer('d_smallest_hidden_unit_n', 64, 'smallest hidden unit num of discriminator')
    tf.app.flags.DEFINE_integer('d_k_size', 3, 'kernel size of discriminator')
    # - Classifier
    tf.app.flags.DEFINE_integer('c_smallest_unit_n', 64, 'smallest hidden unit num of classifier')
    tf.app.flags.DEFINE_integer('c_k_size', 3, 'kernel size of classifier')
    tf.app.flags.DEFINE_float('c_penalty', 0.01, 'learning penalty of classifier')
    tf.app.flags.DEFINE_float('c_lr', 0.0000025, 'learning rate of generator iwth classifier')
    # Train Settings
    tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')
    tf.app.flags.DEFINE_integer('epoch_n', 30, 'epoch cycles')
    tf.app.flags.DEFINE_integer('critic_n', 30, 'how many critic wasserstein distance')
    tf.app.flags.DEFINE_integer('z_size', 100, 'z size')
    # Verbose
    tf.app.flags.DEFINE_integer('save_imgs_interval', 10, 'interval when save imgs')
    tf.app.flags.DEFINE_integer('save_imgs_col_n', 16, 'column num of save imgs')
    tf.app.flags.DEFINE_boolean('is_run_tensorboard', True, 'run tensorboard or not')
    # Source Paths
    tf.app.flags.DEFINE_string('src_real_h5', 'src/fonts_6628_caps_3ch_64x64.h5', 'source path of real fonts hdf5')
    # tf.app.flags.DEFINE_string('src_real_h5', 'src/fonts_200new_caps_3ch_64x64.h5', 'source path of real fonts hdf5')
    tf.app.flags.DEFINE_string('src_classifier_ckpt', 'result_classifier/2017-09-30_180855/log/result_29.ckpt', 'source path of classifier ckpt')
    tf.app.flags.DEFINE_string('src_ckpt', 'result/ckpt/result_9.ckpt', 'source path of result ckpt')
    # Destination Paths
    tf.app.flags.DEFINE_string('dst_root', dst_root, 'destination path')
    tf.app.flags.DEFINE_string('dst_log', dst_root + '/log', 'destination log path')
    tf.app.flags.DEFINE_string('dst_samples', dst_root + '/samples', 'destination samples path')


def main(argv=None):
    if FLAGS.mode == 'train':
        gan = TrainingFontDesignGAN()
        gan.setup()
        gan.train()
    if FLAGS.mode == 'generate':
        gan = GeneratingFontDesignGAN()
        gan.setup()
        gan.generate()


if __name__ == '__main__':
    define_flags()
    tf.app.run()
