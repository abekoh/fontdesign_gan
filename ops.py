from keras import backend as K
from keras.layers.merge import _Merge


def multiple_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def mean_squared_error_inv(y_true, y_pred):
    return K.mean(-K.square(y_pred - y_true), axis=-1)


def hamming_error_inv(y_true, y_pred):
    y_true_signed = K.sign(y_true)
    y_pred_signed = K.sign(y_pred)
    tmp = K.abs(y_pred_signed - y_true_signed) / 2
    return -K.sum(K.sum(tmp, axis=-1), axis=-1)


class Subtract(_Merge):

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output - inputs[i]
        return output
