import numpy as np
import cv2
import os
import operator

def get_power(num, k=0):
    if num <= 2 ** k:
        pad_num = 2 ** k
        return pad_num
    else:
        k += 1
        return get_power(num, k)

def image_padding(img):

    if len(img.shape) == 2 :
        h, w = img.shape[0], img.shape[1]

        h_pad = get_power(h)-h
        w_pad = get_power(w)-w

        img = np.pad(img, pad_width=((0, h_pad), (0, w_pad)), mode='constant')

    elif len(img.shape) == 3:
        h, w = img.shape[0], img.shape[1]

        h_pad = get_power(h) - h
        w_pad = get_power(w) - w
        img = np.pad(img, pad_width=((0, h_pad), (0, w_pad), (0, 0)), mode='constant')

    return img


def DFT_1D(fx):
    fx = np.asarray(fx, dtype=complex)
    M = fx.shape[0]
    fu = fx.copy()

    for i in range(M):
        u = i
        sum = 0
        for j in range(M):
            x = j
            tmp = fx[x]*np.exp(-2j*np.pi*x*u*np.divide(1, M, dtype=complex))
            sum += tmp
        fu[u] = sum

    return fu


def inverseDFT_1D(fu):
    fu = np.asarray(fu, dtype=complex)
    M = fu.shape[0]
    fx = np.zeros(M, dtype=complex)

    for i in range(M):
        x = i
        sum = 0
        for j in range(M):
            u = j
            tmp = fu[u]*np.exp(2j*np.pi*x*u*np.divide(1, M, dtype=complex))
            sum += tmp
        fx[x] = np.divide(sum, M, dtype=complex)

    return fx


def FFT_1D(fx):

    fx = np.asarray(fx, dtype=complex)
    M = fx.shape[0]
    minDivideSize = 4

    if M % 2 != 0:
        raise ValueError("the input size must be 2^n")

    if M <= minDivideSize:
        return DFT_1D(fx)
    else:
        fx_even = FFT_1D(fx[::2])
        fx_odd = FFT_1D(fx[1::2])
        W_ux_2k = np.exp(-2j * np.pi * np.arange(M) / M)

        f_u = fx_even + fx_odd * W_ux_2k[:M//2]

        f_u_plus_k = fx_even + fx_odd * W_ux_2k[M//2:]

        fu = np.concatenate([f_u, f_u_plus_k])

    return fu


def inverseFFT_1D(fu):

    fu = np.asarray(fu, dtype=complex)
    fu_conjugate = np.conjugate(fu)

    fx = FFT_1D(fu_conjugate)

    fx = np.conjugate(fx)
    fx = fx / fu.shape[0]

    return fx


def FFT_2D(fx):
    h, w = fx.shape[0], fx.shape[1]

    fu = np.zeros(fx.shape, dtype=complex)

    if len(fx.shape) == 2:
        for i in range(h):
            fu[i, :] = FFT_1D(fx[i, :])

        for i in range(w):
            fu[:, i] = FFT_1D(fu[:, i])

    elif len(fx.shape) == 3:
        for ch in range(3):
            fu[:, :, ch] = FFT_2D(fx[:, :, ch])

    return fu


def inverseDFT_2D(fu):
    h, w = fu.shape[0], fu.shape[1]

    fx = np.zeros(fu.shape, dtype=complex)

    if len(fu.shape) == 2:
        for i in range(h):
            fx[i, :] = inverseDFT_1D(fu[i, :])

        for i in range(w):
            fx[:, i] = inverseDFT_1D(fx[:, i])

    elif len(fu.shape) == 3:
        for ch in range(3):
            fx[:, :, ch] = inverseDFT_2D(fu[:, :, ch])

    fx = np.real(fx)
    return fx


def inverseFFT_2D(fu):
    h, w = fu.shape[0], fu.shape[1]

    fx = np.zeros(fu.shape, dtype=complex)

    if len(fu.shape) == 2:
        for i in range(h):
            fx[i, :] = inverseFFT_1D(fu[i, :])

        for i in range(w):
            fx[:, i] = inverseFFT_1D(fx[:, i])

    elif len(fu.shape) == 3:
        for ch in range(3):
            fx[:, :, ch] = inverseFFT_2D(fu[:, :, ch])

    fx = np.real(fx)
    return fx

def image_fft(img_origin):
    img = image_padding(img_origin)
    img_fft = FFT_2D(img)
    if len(img_origin.shape) == 2:
        img_fft = img_fft[:img_origin.shape[0], :img_origin.shape[1]]
    else:
        img_fft = img_fft[:img_origin.shape[0], :img_origin.shape[1], :]

    img_fft_complex = img_fft.copy()
    img_fft = np.real(img_fft)

    return img_fft_complex, img_fft


def image_fft_inverse(img_fft_complex):

    img_fft = image_padding(img_fft_complex)
    img_origin = inverseFFT_2D(img_fft)
    img_ifft = np.real(img_origin)
    if len(img_origin.shape) == 2:
        img_ifft = img_ifft[:img_fft_complex.shape[0], :img_fft_complex.shape[1]]
    else:
        img_ifft = img_ifft[:img_fft_complex.shape[0], :img_fft_complex.shape[1], :]

    return img_ifft


if __name__ == '__main__':

    pwd = os.getcwd()
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    image_set_path = father_path + '\origin_images\FFT_images'
    save_results_path = father_path + '\save_image_results\FFT_results'

    def listdir(path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name)
            else:
                list_name.append(file_path)

        return list_name

    image_names = listdir(image_set_path, [])
    for i in range(len(image_names)):
        img_name = os.path.basename(image_names[i])
        print(i, 'processing image: ', img_name)
        file_name = img_name.split('.')[0]
        img_origin = cv2.imread(image_names[i])
        R, G, B = img_origin[:, :, 0], img_origin[:, :, 1], img_origin[:, :, 2]
        if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):  ## is gray
            img_origin = R
        img_fft_complex, img_fft = image_fft(img_origin)
        img_ifft = image_fft_inverse(img_fft_complex)

        fft_save_path = save_results_path + '\\' + str(file_name) + '_fft.png'
        rec_save_path = save_results_path + '\\' + str(file_name) + '_fft_rec.png'

        cv2.imwrite(fft_save_path, img_fft)
        cv2.imwrite(rec_save_path, img_ifft)

    print('End FFT processing.')