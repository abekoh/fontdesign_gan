import numpy as np
from keras import backend as K
from PIL import Image


def show_np_img(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            print(int(img[i][j]), end=' ')
        print()


pil_img1 = Image.open('output/generate/A_0.png')
pil_img2 = Image.open('output/generate/A_2.png')

np_img1 = np.asarray(pil_img1)
np_img2 = np.asarray(pil_img2)

k_img1 = K.variable(np_img1, 'float32', 'img1')
k_img2 = K.variable(np_img2, 'float32', 'img2')

k_result1 = K.mean(K.abs(k_img1 - k_img2))

k_img1_clipped = K.clip(k_img1, 127, 128) - 127
k_img2_clipped = K.clip(k_img2, 127, 128) - 127
k_result2 = K.mean(K.abs(k_img1_clipped - k_img2_clipped))

np_result1 = K.eval(k_result1)
np_result2 = K.eval(k_result2)

print(np_result1)
print(np_result2)
# show_np_img(np_result)
