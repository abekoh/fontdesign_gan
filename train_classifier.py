import os

from keras.optimizers import SGD
from keras.utils import to_categorical, Progbar

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
        self.classifier.compile(optimizer=SGD(lr=0.01, decay=0.0005),
                                loss='categorical_crossentropy', metrics=['accuracy'])

    def load_dataset(self, src_h5_path, is_shuffle=True, train_rate=0.9):
        self.dataset = Dataset(src_h5_path, 'r')
        self.dataset.set_load_data(train_rate=train_rate)
        if is_shuffle:
            self.dataset.shuffle()
        self.train_data_n = self.dataset.get_img_len()
        self.test_data_n = self.dataset.get_img_len(is_test=True)
        print(self.train_data_n, self.test_data_n)

    def train(self, epoch_n=20, batch_size=16):
        train_batch_n = self.train_data_n // batch_size
        test_batch_n = self.test_data_n // batch_size
        for epoch_i in range(epoch_n):
            # train
            progbar = Progbar(train_batch_n)
            losses, accs = list(), list()
            for batch_i in range(train_batch_n):
                progbar.update(batch_i)
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, batch_size)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                loss, acc = self.classifier.train_on_batch(batched_imgs, batched_categorical_labels)
                losses.append(loss)
                accs.append(acc)
            print('[train] loss: {}, acc: {}'.format(sum(losses) / len(losses), sum(accs) / len(accs)))
            # test
            progbar = Progbar(test_batch_n)
            losses, accs = list(), list()
            for batch_i in range(test_batch_n):
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, batch_size, is_test=True)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                loss, acc = self.classifier.test_on_batch(batched_imgs, batched_categorical_labels)
                losses.append(loss)
                accs.append(acc)
            print('[test] loss: {}, acc: {}'.format(sum(losses) / len(losses), sum(accs) / len(accs)))
        self.classifier.save_weights(os.path.join(self.output_root_dir_path, 'classifier.h5'))

    def _labels_to_categorical(self, labels):
        return to_categorical(list(map(lambda x: ord(x) - 65, labels)), 26)


if __name__ == '__main__':
    cl = TrainingClassifier()
    cl.build_models()
    cl.load_dataset('./fonts_selected_200_eng.h5')
    cl.train(epoch_n=20)
