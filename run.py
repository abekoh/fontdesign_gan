from datetime import datetime
from keras.optimizers import Adam, RMSprop

from train_gan import TrainingFontDesignGAN
from params import Params

if __name__ == '__main__':
    params = Params(d={
        'img_size': (256, 256),
        'img_dim': 1,
        'font_embedding_n': 200,
        'char_embedding_n': 26,
        'epoch_n': 50,
        'batch_size': 16,
        'critic_n': 5,
        # 'early_stopping_n': 10,
        'save_metrics_graph_interval': 1,
        'save_metrics_smoothing_graph_interval': 10,
        'save_imgs_interval': 10,
        'save_weights_interval': 5,
        'g': Params({
            'arch': 'pix2pix',
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [1.]
        }),
        'd': Params({
            'arch': 'pix2pix',
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [1.]
        }),
        'dc': Params({
            'opt': Adam(lr=0.0002, beta_1=0.5),
            'loss_weights': [0.5],
        }),
        'gc': Params({
            'opt': Adam(lr=0.0002, beta_1=0.5),
            'loss_weights': [1.],
        }),
        # 'c': Params({
        #     'opt': RMSprop(lr=0.00005),
        #     'loss_weights': [0.5]
        # }),
        'l1': Params({
            'opt': Adam(lr=0.0002, beta_1=0.5),
            'loss_weights': [100.]
        }),
        # 'v': Params({
        #     'opt': RMSprop(lr=0.00005),
        #     'loss_weights': [1.]
        # }),
        'e': Params({
            'opt': Adam(lr=0.0002, beta_1=0.5),
            'loss_weights': [15.]
        })
    })

    str_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    dst_root = 'output/' + str_now

    paths = Params({
        'src': Params({
            'real_h5': 'src/fonts_200_caps_256x256.h5',
            'src_h5': 'src/arial.h5',
            # 'cls_weight_h5': 'output_classifier/classifier_weights_20(train=0.936397172634403,test=0.9258828996282528).h5'
        }),
        'dst': Params({
            'root': dst_root,
            'tensorboard_log': '{}/tensorboard_log'.format(dst_root),
            'generated_imgs': '{}/generated_imgs'.format(dst_root),
            'model_weights': '{}/model_weights'.format(dst_root),
            'metrics': '{}/metrics'.format(dst_root),
        })
    })
    gan = TrainingFontDesignGAN(params, paths)
    gan.train()
