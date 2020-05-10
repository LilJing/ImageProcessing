import cv2
import numpy as np
import os

def DCT_sub(sub_img):
    C_temp = np.zeros(sub_img.shape)

    m, n = sub_img.shape
    N = n
    C_temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)
                                  ) * np.sqrt(2 / N)

    dst = np.dot(C_temp, sub_img)
    dst = np.dot(dst, np.transpose(C_temp))

    dst_sub = np.log(abs(dst))

    img_recor = np.dot(np.transpose(C_temp), dst)
    img_recor_sub = np.dot(img_recor, C_temp)

    return dst_sub, img_recor_sub

def DCT(img):

    if len(img.shape) == 3:
        B, G, R = cv2.split(img)
        H = B.shape[0]
        W = B.shape[1]
        h = 8
        w = 8
        num_h = int(np.floor(H / h))
        num_w = int(np.floor(W / w))

        num = 0
        img_new_dct_B, img_new_rec_B = np.zeros((H, W)), np.zeros((H, W))
        img_new_dct_G, img_new_rec_G = np.zeros((H, W)), np.zeros((H, W))
        img_new_dct_R, img_new_rec_R = np.zeros((H, W)), np.zeros((H, W))

        for i in range(num_h):
            for j in range(num_w):
                num += 1
                sub_B = B[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))]
                sub_G = G[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))]
                sub_R = R[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))]

                dst_sub_B, img_recor_sub_B = DCT_sub(sub_B)
                dst_sub_G, img_recor_sub_G = DCT_sub(sub_G)
                dst_sub_R, img_recor_sub_R = DCT_sub(sub_R)

                img_new_dct_B[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))] = dst_sub_B
                img_new_rec_B[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))] = img_recor_sub_B
                img_new_dct_G[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))] = dst_sub_G
                img_new_rec_G[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))] = img_recor_sub_G
                img_new_dct_R[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))] = dst_sub_R
                img_new_rec_R[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))] = img_recor_sub_R

        img_new_dct = cv2.merge([img_new_dct_B, img_new_dct_G, img_new_dct_R])
        img_new_rec = cv2.merge([img_new_rec_B, img_new_rec_G, img_new_rec_R])


    else:
        H = img.shape[0]
        W = img.shape[1]
        h = 8
        w = 8
        num_h = int(np.floor(H / h))
        num_w = int(np.floor(W / w))
        print('num_h', num_h, 'num_w', num_w)
        num = 0
        img_new_dct, img_new_rec = np.zeros((H, W)), np.zeros((H, W))
        for i in range(num_h):
            for j in range(num_w):
                num += 1
                sub = img[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))]
                dst_sub, img_recor_sub = DCT_sub(sub)
                img_new_dct[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))] = dst_sub
                img_new_rec[(h * i): (h * (i + 1)), (w * j): (w * (j + 1))] = img_recor_sub

    return img_new_dct, img_new_rec



if __name__ == '__main__':

    pwd = os.getcwd()
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    image_set_path = father_path + '\origin_images\DCT_images'
    save_results_path = father_path + '\save_image_results\DCT_results'

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
        img = cv2.imread(image_names[i])
        img_new_dct, img_new_rec = DCT(img)
        img_new_dct = img_new_dct.astype(np.uint8)
        img_new_rec = img_new_rec.astype(np.uint8)

        dct_save_path = save_results_path + '\\' + str(file_name) + '_dct.jpg'
        rec_dct_save_path = save_results_path + '\\' + str(file_name) + '_dct_rec.jpg'

        cv2.imwrite(dct_save_path, img_new_dct)
        cv2.imwrite(rec_dct_save_path, img_new_rec)

    print('End DCT processing.')