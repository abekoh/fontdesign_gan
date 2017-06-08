from PIL import Image
from models import Generator
from keras.layers import Dense
from keras.layers import Conv2D
import numpy as np
from keras.utils import np_utils

CAPS = [chr(i) for i in range(65, 65 + 26)]

if __name__ == '__main__':
    imgs = np.empty((0, 256, 256, 3), np.float32)
    labels = np.empty((0, 1), int)
    for c in CAPS:
        img_pil = Image.open('src/{}.png'.format(c))
        img_rgb = Image.new('RGB', img_pil.size)
        img_rgb.paste(img_pil)
        img_np = np.asarray(img_rgb)
        img_np = img_np.astype(np.float32) / 255.0
        img_np = img_np[np.newaxis, :, :, :]
        imgs = np.append(imgs, img_np, axis=0)
        labels = np.append(labels, np.array([[ord(c) - 65]]), axis=0)
    # imgs = imgs[:, :, :, np.newaxis]
    labels = np_utils.to_categorical(labels, 26)

    # base_encoder = Encoder()
    # for i in range(1, 9):
    #     intermediate_encoder = Model(base_encoder.input, base_encoder.get_layer('en_l{}'.format(i)).output)
    #     res = intermediate_encoder.predict_on_batch(imgs)
    #     print(res.shape)
    #
    # base_decoder = Decoder(base_encoder)
    ids = np.array([i for i in range(26)])
    generator = Generator()
    res = generator.predict_on_batch([imgs, ids])
    print(res.shape)
