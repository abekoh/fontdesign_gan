from datetime import datetime
from keras.optimizers import RMSprop
from keras.initializers import truncated_normal
from hyperopt import fmin, hp, rand

from train_gan import TrainingFontDesignGAN
from params import Params


def run_train_gan(args):
    print(args)

    params = Params(d={
        'img_size': (128, 128),
        'img_dim': 1,
        'font_embedding_n': 40,
        'char_embedding_n': 26,
        'epoch_n': 10,
        'batch_size': args['batch_size']
        'critic_n': 2,
        # 'early_stopping_n': 10,
        'save_metrics_graph_interval': 1,
        'save_metrics_smoothing_graph_interval': 10,
        'save_imgs_interval': 10,
        'save_weights_interval': 5,
        'is_auto_open': True,
        'g': Params({
            'arch': 'dcgan',
            'layer_n': 4,
            'smallest_hidden_unit_n': 128,
            'kernel_initializer': truncated_normal(),
            'is_bn': True,
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [1.]
        }),
        'd': Params({
            'arch': 'dcgan',
            'layer_n': 4,
            'smallest_hidden_unit_n': 128,
            'kernel_initializer': truncated_normal(),
            'is_bn': args['d.is_bn'],
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [1.]
        }),
        'c': Params({
            'opt': RMSprop(lr=0.00005),
            'loss_weights': [args['c.loss_weights']]
        })
    })

    str_now = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    dst_root = 'output_search_hyper_params/' + str_now

    paths = Params({
        'src': Params({
            'real_h5': 'src/fonts_200_caps_128x128.h5',
            'src_h5': 'src/arial_128x128.h5',
            'cls_weight_h5': 'output_classifier/classifier_128x128_weights_10(train=0.9359328242699412,test=0.9194934944237918).h5'
        }),
        'dst': Params({
            'root': dst_root,
            'tensorboard_log': '{}/tensorboard_log'.format(dst_root),
            'generated_imgs': '{}/generated_imgs'.format(dst_root),
            'model_weights': '{}/model_weights'.format(dst_root),
            'metrics': '{}/metrics'.format(dst_root),
            'model_visualization': '{}/model_visualization'.format(dst_root)
        })
    })

    gan = TrainingFontDesignGAN(params, paths)
    gan.train()
    return gan.get_last_metric('d_wasserstein')


space = {
    'batch_size:' hp.choice('batch_size', [16, 32, 64]),
    'd.is_bn': hp.choice('d.is_bn', [True, False]),
    'c.loss_weights': hp.uniform('c.loss_weights', 0.01, 0.50)
}

if __name__ == '__main__':
    best = fmin(run_train_gan, space, algo=rand.suggest, max_evals=5)
    print(best)
