import os
import json

from keras.utils import to_categorical, plot_model
from tqdm import tqdm

from models import ClassifierMin, Classifier
from dataset import Dataset


class TrainingClassifier():

    def __init__(self, params, paths):
        self.params = params
        self.paths = paths
        self._set_outputs()
        self._save_params()
        self._build_models()
        self._load_dataset()

    def _set_outputs(self):
        if not os.path.exists(self.paths.dst.root):
            os.makedirs(self.paths.dst.root)

    def _save_params(self):
        with open(os.path.join(self.paths.dst.root, 'params.txt'), 'w') as f:
            json.dump(self.params.to_dict(), f, indent=4)
        with open(os.path.join(self.paths.dst.root, 'paths.txt'), 'w') as f:
            json.dump(self.paths.to_dict(), f, indent=4)

    def _build_models(self):
        self.classifier = Classifier(img_size=(self.params.img_size[0], self.params.img_size[1]), img_dim=self.params.img_dim, class_n=26)
        self.classifier.compile(optimizer=self.params.opt,
                                loss='categorical_crossentropy', metrics=['accuracy'])
        plot_model(self.classifier, to_file=os.path.join(self.paths.dst.root, 'model.png'), show_shapes=True)
        with open(os.path.join(self.paths.dst.root, 'model.json'), 'w') as json_file:
            json_dict = self.classifier.to_json()
            json.dump(json_dict, json_file)

    def _load_dataset(self):
        self.dataset = Dataset(self.paths.src.fonts, 'r', self.params.img_size, img_dim=self.params.img_dim)
        self.dataset.set_load_data(train_rate=self.params.train_rate)
        if self.params.is_shuffle:
            self.dataset.shuffle()
        self.train_data_n = self.dataset.get_img_len()
        self.test_data_n = self.dataset.get_img_len(is_test=True)

    def train(self):
        train_batch_n = self.train_data_n // self.params.batch_size
        test_batch_n = self.test_data_n // self.params.batch_size
        for epoch_i in tqdm(range(self.params.epoch_n)):
            # train
            losses, accs = list(), list()
            for batch_i in tqdm(range(train_batch_n)):
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, self.params.batch_size)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                loss, acc = self.classifier.train_on_batch(batched_imgs, batched_categorical_labels)
                losses.append(loss)
                accs.append(acc)
            train_loss_avg = sum(losses) / len(losses)
            train_acc_avg = sum(accs) / len(accs)
            print('[train] loss: {}, acc: {}'.format(train_loss_avg, train_acc_avg))
            # test
            losses, accs = list(), list()
            for batch_i in tqdm(range(test_batch_n)):
                batched_imgs, batched_labels = self.dataset.get_batch(batch_i, self.params.batch_size, is_test=True)
                batched_categorical_labels = self._labels_to_categorical(batched_labels)
                loss, acc = self.classifier.test_on_batch(batched_imgs, batched_categorical_labels)
                losses.append(loss)
                accs.append(acc)
            test_loss_avg = sum(losses) / len(losses)
            test_acc_avg = sum(accs) / len(accs)
            print('[test] loss: {}, acc: {}\n'.format(test_loss_avg, test_acc_avg))
            if (epoch_i + 1) % self.params.save_weights_interval == 0 or epoch_i + 1 == self.params.epoch_n:
                weights_filename = 'classifier_weights_{0}(train={1:03.3f},test={2:03.3f}).h5'.format(epoch_i + 1, train_acc_avg, test_acc_avg)
                self.classifier.save_weights(os.path.join(self.paths.dst.root, weights_filename))

    def _labels_to_categorical(self, labels):
        return to_categorical(list(map(lambda x: ord(x) - 65, labels)), 26)
