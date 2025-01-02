import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn

X = misc.ascent()
X_compressed = np.zeros(X.shape)

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

nblocks_w = X.shape[1] // 8
nblocks_h = X.shape[0] // 8

for i in range(nblocks_h):
    for j in range(nblocks_w):
        block = X[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
        coefs_block = dctn(block)
        encoded_block = coefs_block // Q_jpeg
        decoded_block = encoded_block * Q_jpeg
        compressed_block = idctn(decoded_block)
        X_compressed[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = compressed_block

X_compressed = np.round(X_compressed)
X_compressed = X_compressed.astype('uint8')

plt.figure()
plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(X_compressed, cmap=plt.cm.gray)
plt.title('JPEG')
plt.show()