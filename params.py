from keras.optimizers import Optimizer
from keras.initializers import Initializer


class Params():

    def __init__(self, d=None):
        if d:
            self.from_dict(d)

    def from_dict(self, d):
        for attr, value in d.items():
            setattr(self, attr, value)

    def to_dict(self):
        d = dict()
        for attr, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                d[attr] = value.to_dict()
            elif isinstance(value, Optimizer) or isinstance(value, Initializer):
                d[attr] = '{} {}'.format(value, value.get_config())
            else:
                d[attr] = value
        return d
