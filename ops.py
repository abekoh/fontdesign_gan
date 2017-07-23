from keras import backend as K


def mean_only(y_true, y_pred):
    return K.mean(y_true, axis=-1)


def wasserstein_distance(y_true, y_pred):
    return K.mean(y_true - y_pred, axis=-1)
