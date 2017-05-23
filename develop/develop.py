from PIL import Image
from models import Encoder, Decoder
import numpy as np
from keras.utils import np_utils
from keras.layers import Input

CAPS = [chr(i) for i in range(65, 65 + 26)]

if __name__ == '__main__':
    imgs = np.empty((0, 256, 256), np.float32)
    labels = np.empty((0, 1), int)
    for c in CAPS:
        img_pil = Image.open('src/{}.png'.format(c))
        img_np = np.asarray(img_pil)
        img_np = img_np.astype(np.float32) / 255.0
        img_np = img_np[np.newaxis, :, :]
        imgs = np.append(imgs, img_np, axis=0)
        labels = np.append(labels, np.array([[ord(c) - 65]]), axis=0)
    imgs = imgs[:, :, :, np.newaxis]
    labels = np_utils.to_categorical(labels, 26)

    encoder = Encoder()
    res = encoder.predict_on_batch(imgs)
    print (res.shape)
    # # decoder = Decoder(encoder)
    # generator = Generator(encoder)
    # # encoder.compile(optimizer='adam', loss='categorical_crossentropy')
    # # result =  encoder.predict_on_batch(imgs)
    # id_array = np.array([1])
    # # id_array = id_array[np.newaxis, np.newaxis, :]
    # result = generator.predict_on_batch(imgs, id_array)
    # print (imgs.shape)
    # print (result.shape)
