class Params():

    def __init__(self, d=None):
        if d:
            self.from_dict(d)

    def from_dict(self, d):
        for attr, value in d.items():
            if hasattr(self, attr):
                setattr(self, attr, value)


class ModelParams(Params):

    def __init__(self):
        self.arch = ''
        self.lr = 0.001
        self.clipvalue = 0.01
        self.loss_weights = [1.]


class GANParams(Params):

    def __init__(self, d=None):
        super(GANParams, self).__init__(d)
        self.img_dim = 1
        self.embedding_n = 40
        self.epoch_n = 30
        self.batch_size = 32
        self.critic_n = 5
        self.early_stopping_n = 10


class GANPaths(Params):

    def __init__(self):
        self.dst_dir_path = 'output_gan'
        self.src_real_h5_path = 'src/fonts_200_caps_256x256.h5'
        self.src_src_h5_path = 'src/arial.h5'
