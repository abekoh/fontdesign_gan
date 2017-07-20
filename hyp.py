from train_gan import TrainingFontDesignGAN
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform


def data():
    return 1


def model(dummy):
    gan = TrainingFontDesignGAN()
    gan.build_models(
        classifier_h5_path='output_classifier/classifier_weights_20(train=0.936397172634403,test=0.9258828996282528).h5',
        lr={{uniform(0.00001, 0.0002)}},
        beta_1={{uniform(0.1, 0.9)}},
        loss_weights={'d': [{{uniform(1., 20.)}}],
                      'g2d': [{{uniform(1., 20.)}}],
                      'g2e': [{{uniform(1., 20.)}}],
                      'g2c': [{{uniform(1., 20.)}}]})
    gan.load_dataset('src/fonts_selected_200_eng.h5', 'src/arial.h5')
    sum_loss = gan.train(epoch_n=1)
    return {'loss': sum_loss, 'status': STATUS_OK, 'model': gan.get_generator_model()}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
