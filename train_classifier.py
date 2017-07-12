import os
import numpy as np

from keras.optimizers import Adam
from keras.utils import to_categorical

from models import Classifier
from dataset import Dataset


class TrainingClassifier():

    def __init__(self, output_dir_path='output_classifier'):
        self._set_outputs(output_root_dir_path=output_dir_path)

    def _set_outputs(self, output_root_dir_path):
        self.output_root_dir_path = output_root_dir_path
        if not os.path.exists(self.output_root_dir_path):
            os.mkdir(self.output_root_dir_path)

    def build_models(self, img_dim=1, class_n=26):
        self.img_dim = img_dim
        self.class_n = class_n

        self.classifier = Classifier(img_dim=img_dim, class_n=class_n)
        self.classifier.compile(optimizer=Adam(),
                                loss='categorical_crossentropy', metrics=['accuracy'])

    def load_dataset(self, src_h5_path, is_shuffle=True):
        dataset = Dataset()
        dataset.load_h5_for_classifier(src_h5_path)
        self.train_imgs, self.train_labels, self.test_imgs, self.test_labels = dataset.get_for_classifier()

    def _shuffle_dataset(self, imgs, labels):
        combined = np.c_[imgs.reshape(imgs.shape[0], -1), labels]
        np.random.shuffle(combined)
        imgs_n = imgs.size // imgs.shape[0]
        imgs = combined[:, :imgs_n].reshape(imgs.shape)
        labels = combined[:, imgs_n:].reshape(labels.shape)
        return imgs, labels

    def train(self, epoch_n=20, batch_size=16):

        self.classifier.fit(self.train_imgs, to_categorical(self.train_labels, 26), batch_size=16, epochs=epoch_n)

        loss, acc = self.classifier.evaluate(self.test_imgs, to_categorical(self.test_labels, 26), batch_size=16)

        print('accuracy: {}'.foramt(acc))

        self.classifier.save_weights(os.path.join(self.output_root_dir_path, 'classifier_h5'))


if __name__ == '__main__':
    cl = TrainingClassifier()
    cl.build_models()
    cl.load_dataset('./font_200_selected_alphs_for_classifier.h5')
    cl.train(epoch_n=1)
