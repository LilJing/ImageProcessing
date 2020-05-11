import numpy as np
import cv2
import os
import operator

def RGB2HSI(rgb_img):
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]

    hsi_img = rgb_img.copy()

    B,G,R = cv2.split(rgb_img)

    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    H = np.zeros((row, col))
    I = (R + G + B) / 3.0
    S = np.zeros((row,col))
    for i in range(row):
        den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
        thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)
        h = np.zeros(col)

        h[B[i]<=G[i]] = thetha[B[i]<=G[i]]

        h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]

        h[den == 0] = 0
        H[i] = h/(2*np.pi)

    for i in range(row):
        min = []
        for j in range(col):
            arr = [B[i][j],G[i][j],R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        S[i] = 1 - min*3/(R[i]+B[i]+G[i])
        S[i][R[i]+B[i]+G[i] == 0] = 0

    hsi_img[:,:,0] = H*255
    hsi_img[:,:,1] = S*255
    hsi_img[:,:,2] = I*255
    return hsi_img

def createEmptyImage(rows, cols, type):
    img = np.zeros((rows, cols), dtype=type)
    return img

def createEmptyList(size):
    newList = []
    for eachNum in range(0, size):
        newList.append(0)
    return newList

def histequaLize(src, dst):

    hist = createEmptyList(256)
    rows, cols = src.shape
    for r in range(rows):
        for c in range(cols):
            hist[src[r, c]] += 1

    for i in range(256):
        hist[i] /= rows * cols
#
    for i in range(1, 256):
        hist[i] = hist[i - 1] + hist[i]

    for i in range(256):
        hist[i] = (np.uint8)(255 * hist[i] + 0.5)

    for r in range(rows):
        for c in range(cols):
            dst[r, c] = hist[src[r, c]]

def Histogram_Color(image):
    hsi_img = RGB2HSI(image)

    B, G, R = cv2.split(image)
    dst1 = createEmptyImage(B.shape[0], B.shape[1], np.uint8)
    histequaLize(B, dst1)
    dst2 = createEmptyImage(G.shape[0], G.shape[1], np.uint8)
    histequaLize(G, dst2)
    dst3 = createEmptyImage(R.shape[0], R.shape[1], np.uint8)
    histequaLize(R, dst3)
    img_hist = cv2.merge([dst1, dst2, dst3])

    hsi_B, hsi_G, hsi_R = cv2.split(hsi_img)
    hsi_dst1 = createEmptyImage(hsi_B.shape[0], hsi_B.shape[1], np.uint8)
    histequaLize(hsi_B, hsi_dst1)
    hsi_dst2 = createEmptyImage(hsi_G.shape[0], hsi_G.shape[1], np.uint8)
    histequaLize(hsi_G, hsi_dst2)
    hsi_dst3 = createEmptyImage(hsi_R.shape[0], hsi_R.shape[1], np.uint8)
    histequaLize(hsi_R, hsi_dst3)
    hsi_img_hist = cv2.merge([hsi_dst1, hsi_dst2, hsi_dst3])

    return hsi_img, img_hist, hsi_img_hist

def Histogram_gray(img):

    dst = createEmptyImage(img.shape[0], img.shape[1], np.uint8)
    histequaLize(img, dst)

    return dst

if __name__ == '__main__':

    pwd = os.getcwd()
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    image_set_path = father_path + '\origin_images\Histogram_images'
    save_results_path = father_path + '\save_image_results\Histogram_results'

    def listdir(path, list_name):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name)
            else:
                list_name.append(file_path)

        return list_name

    image_names = listdir(image_set_path, [])
    print('images number', len(image_names))
    for i in range(len(image_names)):
        img_name = os.path.basename(image_names[i])
        print(i, 'processing image: ', img_name)
        file_name = img_name.split('.')[0]
        image = cv2.imread(image_names[i])
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):  ## is gray
            image = R
            img_hist = Histogram_gray(image)
            hist_save_path = save_results_path + '\\' + str(file_name) + '_histogram.jpg'
            cv2.imwrite(hist_save_path, img_hist)
        else:
        # if len(image.shape) == 3:
            hsi_img, img_hist, hsi_img_hist = Histogram_Color(image)
            hsi_save_path = save_results_path + '\\' + str(file_name) + '_hsi.jpg'
            hist_save_path = save_results_path + '\\' + str(file_name) + '_histogram.jpg'
            hsi_hist_save_path = save_results_path + '\\' + str(file_name) + '_hsi_histgram.jpg'

            cv2.imwrite(hsi_save_path, hsi_img)
            cv2.imwrite(hist_save_path, img_hist)
            cv2.imwrite(hsi_hist_save_path, hsi_img_hist)

    print('End Histogram Equalization processing.')
