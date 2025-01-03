import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
import cv2 as cv

# constants

# TODO remove hardcoded path
video_path = './videos/original.mp4'
output_path = './videos/compressed.mp4'

Q_luminance = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

Q_chrominance = [[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]]

RGB_to_YCbCr = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
YCbCr_to_RGB = np.array([[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]])

def process_image(frame):
    X = frame.copy()
    X_YCbCr = np.zeros(X.shape)
    X_compressed = np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_YCbCr[i,j,:] = RGB_to_YCbCr @ X[i,j,:]
                    
    X_YCbCr[:,:,1] += 128
    X_YCbCr[:,:,2] += 128

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(3):
                X_YCbCr[i,j,k] = min(max(0, np.round(X_YCbCr[i,j,k])),255)

    X_YCbCr = X_YCbCr.astype('uint8')

    Y_channel = X_YCbCr[:,:,0]
    Cb_channel = X_YCbCr[:,:,1]
    Cr_channel = X_YCbCr[:,:,2]

    # 2x2 means for the Cb and Cr channels

    aux_Cb = np.zeros((Cb_channel.shape[0] // 2, Cb_channel.shape[1] // 2))
    aux_Cr = np.zeros((Cr_channel.shape[0] // 2, Cr_channel.shape[1] // 2))

    for i in range(0, Cb_channel.shape[0], 2):
        for j in range(0, Cb_channel.shape[1], 2):
            mean_patch = np.mean(Cb_channel[i:i + 2,j:j + 2])
            aux_Cb[i // 2, j // 2] = mean_patch
            
    for i in range(0, Cr_channel.shape[0], 2):
        for j in range(0, Cr_channel.shape[1], 2):
            mean_patch = np.mean(Cr_channel[i:i + 2,j:j + 2])
            aux_Cr[i // 2, j // 2] = mean_patch

    Cb_channel = aux_Cb
    Cr_channel = aux_Cr

    Cb_channel = np.round(Cb_channel)
    Cb_channel = Cb_channel.astype('uint8')

    Cr_channel = np.round(Cr_channel)
    Cr_channel = Cr_channel.astype('uint8')

    # JPEG Encoder
    # Forward DCT

    # Y channel
    Y_compressed = np.zeros(Y_channel.shape)

    nblocks_w = Y_compressed.shape[1] // 8
    nblocks_h = Y_compressed.shape[0] // 8

    for i in range(nblocks_h):
        for j in range(nblocks_w):
            block = Y_channel[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            coefs_block = dctn(block)
            encoded_block = np.round(coefs_block / Q_luminance)
            decoded_block = encoded_block * Q_luminance
            compressed_block = idctn(decoded_block)
            Y_compressed[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = compressed_block
            
    Y_compressed = np.round(Y_compressed)
    Y_compressed = Y_compressed.astype('uint8')

    # Cb channel
    Cb_compressed = np.zeros(Cb_channel.shape)

    nblocks_w = Cb_channel.shape[1] // 8
    nblocks_h = Cb_channel.shape[0] // 8

    for i in range(nblocks_h):
        for j in range(nblocks_w):
            block = Cb_channel[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            coefs_block = dctn(block)
            encoded_block = np.round(coefs_block / Q_chrominance)
            decoded_block = encoded_block * Q_chrominance
            compressed_block = idctn(decoded_block)
            Cb_compressed[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = compressed_block
            
    Cb_compressed = np.round(Cb_compressed)
    Cb_compressed = Cb_compressed.astype('uint8')

    # Cr channel
    Cr_compressed = np.zeros(Cr_channel.shape)

    nblocks_w = Cr_channel.shape[1] // 8
    nblocks_h = Cr_channel.shape[0] // 8

    for i in range(nblocks_h):
        for j in range(nblocks_w):
            block = Cr_channel[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            coefs_block = dctn(block)
            encoded_block = np.round(coefs_block / Q_chrominance)
            decoded_block = encoded_block * Q_chrominance
            compressed_block = idctn(decoded_block)
            Cr_compressed[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = compressed_block
            
    Cr_compressed = np.round(Cr_compressed)
    Cr_compressed = Cr_compressed.astype('uint8')

    # we now have Y_compressed, Cb_compressed, Cr_compressed, it is time to put them back together

    # resize the Cb and Cr channels to Y channel dimensions

    Cb_compressed_resized = np.zeros((Cb_compressed.shape[0] * 2, Cb_compressed.shape[1] * 2))
    Cr_compressed_resized = np.zeros((Cr_compressed.shape[0] * 2, Cr_compressed.shape[1] * 2))

    for i in range(0, Cb_compressed_resized.shape[0]):
        for j in range(0, Cb_compressed_resized.shape[1]):
            Cb_compressed_resized[i,j] = Cb_compressed[i // 2, j // 2]
            
    for i in range(0, Cr_compressed_resized.shape[0]):
        for j in range(0, Cr_compressed_resized.shape[1]):
            Cr_compressed_resized[i,j] = Cr_compressed[i // 2, j // 2]

    # make the recovered X_ycbcr_compressed array

    X_YCbCr_compressed = np.stack((Y_compressed, Cb_compressed_resized, Cr_compressed_resized), axis = -1)

    X_YCbCr_compressed[:,:,1] -= 128
    X_YCbCr_compressed[:,:,2] -= 128

    for i in range(X_YCbCr_compressed.shape[0]):
        for j in range(X_YCbCr_compressed.shape[1]):
            X_compressed[i,j,:] = YCbCr_to_RGB @ X_YCbCr_compressed[i,j,:]
            for k in range(3):
                X_compressed[i,j,k] = min(max(0, np.round(X_compressed[i,j,k])),255)
            
    X_compressed = X_compressed.astype('uint8')
    
    return X_compressed


# read the video file
video_cv = cv.VideoCapture(video_path)

if not video_cv.isOpened():
    print("Error: Cannot open video.")

frames = []

while True:
    ret, frame = video_cv.read()
    if not ret:
        print("End of video.")
        break
    
    frames.append(frame)

video_cv.release()

#frames = np.array(frames)
#print(frames)
#compressed_frames = np.zeros(frames.shape)

# process each frame
# for (idx, frame) in enumerate(frames):
#     compressed_frames[idx] = process_image(frame)
#     print(f"Frame {idx}/{compressed_frames.shape[0]} processed")

# output the video

video_output_cv = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1], frames[0].shape[0]))

for frame in frames:
    video_output_cv.write(frame)

video_output_cv.release()