from keras import backend as K
from keras.layers.merge import _Merge


def multiple_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def mean_squared_error_inv(y_true, y_pred):
    return -K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error_inv(y_true, y_pred):
    return -K.mean(K.abs(y_pred - y_true), axis=-1)


class Subtract(_Merge):

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output - inputs[i]
        return output
