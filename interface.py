import tkinter
import tkinter.filedialog
from PIL import Image, ImageTk
import operator

from histogram import *
from fft import *
from dct import *

win = tkinter.Tk()
win.title("Image Display")
win.geometry("1280x1080")

count = 0
img2 = tkinter.Label(win)


def resize(w, h, w_box, h_box, pil_image):

    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])

    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


# size of image display box we want
w_box = 512
h_box = 512


def choose_file():
    select_file = tkinter.filedialog.askopenfilename(title='选择图片')
    e.set(select_file)
    load = Image.open(select_file)
    load = load.convert("RGB")
    w, h = load.size
    pil_image_resized = resize(w, h, w_box, h_box, load)
    global original
    original = pil_image_resized
    render = ImageTk.PhotoImage(original)
    img = tkinter.Label(win, image=render)
    img.image = render
    img.place(x=10, y=100)


def Histogram_():
    print('Begin histogram processing ...')
    temp = original
    img_np = np.asarray(temp)
    # check if color
    R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):
        # print('This is a gray image')
        img_np = R
        new_im_np = Histogram_gray(img_np)
        new_im = Image.fromarray(new_im_np.astype('uint8')).convert('L')
    else:
        # print('This is a color image')
        _, new_im_np, _ = Histogram_Color(img_np)
        new_im = Image.fromarray(new_im_np.astype('uint8')).convert('RGB')

    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=750, y=100)
    print('End histogram equalization of origin image.')


def HSI_():
    print('Begin hsi processing ...')
    temp = original
    img_np = np.asarray(temp)
    # check if color
    R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):
        raise ValueError("The input of HSI must be color image!")
    else:
        new_im_np, _, _ = Histogram_Color(img_np)
        new_im = Image.fromarray(new_im_np.astype('uint8')).convert('RGB')
    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=750, y=100)
    print('End HSI transformation of origin image.')


def HSI_Histogram_():
    print('Begin hsi histogram processing ...')
    temp = original
    img_np = np.asarray(temp)
    # check if color
    R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):
        raise ValueError("The input of HSI_Histogram must be color image!")
    else:
        _ , _, new_im_np = Histogram_Color(img_np)
        new_im = Image.fromarray(new_im_np.astype('uint8')).convert('RGB')

    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=750, y=100)
    print('End histogram equalization of HSI image.')

def FFT_():
    print('Begin fft processing ...')
    temp = original
    img_np = np.asarray(temp)
    # check if color
    R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):
        # print('This is a gray image')
        img_np = R
        _, img_fft = image_fft(img_np)
        new_im = Image.fromarray(img_fft.astype('uint8')).convert('L')
    else:
        # print('This is a color image')
        _, img_fft = image_fft(img_np)
        new_im = Image.fromarray(img_fft.astype('uint8')).convert('RGB')

    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=750, y=100)
    print('End fft of origin image.')

def IFFT_():
    print('Begin inverse fft processing ...')
    temp = original
    img_np = np.asarray(temp)
    ## check if color
    R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):
        # print('This is a gray image')
        img_np = R
        img_fft_complex, img_fft = image_fft(img_np)
        img_ifft = image_fft_inverse(img_fft_complex)
        new_im = Image.fromarray(img_ifft)
    else:
        # print('This is a color image')
        img_fft_complex, img_fft = image_fft(img_np)
        img_ifft = image_fft_inverse(img_fft_complex)
        new_im = Image.fromarray(img_ifft.astype('uint8')).convert('RGB')

    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=750, y=100)
    print('End inverse fft of origin image.')


def DCT_():
    print('Begin dct processing ...')
    temp = original
    img_np = np.asarray(temp)
    # check if color
    R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):
        # print('This is a gray image')
        img_np = R
        img_new_dct, _ = DCT(img_np)
        new_im = Image.fromarray(img_new_dct.astype('uint8')).convert('L')
    else:
        # print('This is a color image')
        img_new_dct, _ = DCT(img_np)
        new_im = Image.fromarray(img_new_dct.astype('uint8')).convert('RGB')

    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=750, y=100)
    print('End dct of origin image.')


def IDCT_():
    print('Begin inverse dct processing ...')
    temp = original
    img_np = np.asarray(temp)
    # check if color
    R, G, B = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    if operator.eq(R.tolist(), G.tolist()) and operator.eq(G.tolist(), B.tolist()):
        # print('This is a gray image')
        img_np = R
        _, img_new_rec = DCT(img_np)
        new_im = Image.fromarray(img_new_rec.astype('uint8')).convert('L')
    else:
        # print('This is a color image')
        _, img_new_rec = DCT(img_np)
        print('img_new_rec', img_new_rec.shape)
        new_im = Image.fromarray(img_new_rec.astype('uint8')).convert('RGB')
        cv2.imwrite('dct_test.png', img_new_rec)

    render = ImageTk.PhotoImage(new_im)
    global img2
    img2.destroy()
    img2 = tkinter.Label(win, image=render)
    img2.image = render
    img2.place(x=750, y=100)
    print('End inverse dct of origin image.')



e = tkinter.StringVar()
e_entry = tkinter.Entry(win, width=68, textvariable=e)
e_entry.pack()

label1 = tkinter.Label(win, text="Original Image")
label1.place(x=200, y=50)

label2 = tkinter.Label(win, text="New Image")
label2.place(x=900, y=50)

button1 = tkinter.Button(win, text="Select", command=choose_file)
button1.place(x=600, y=100)

button2 = tkinter.Button(win, text="Origin Histogram", command=Histogram_)
button2.place(x=600, y=150)

button3 = tkinter.Button(win, text="Convert to HSI", command=HSI_)
button3.place(x=600, y=200)

button4 = tkinter.Button(win, text="HSI Histogram", command=HSI_Histogram_)
button4.place(x=600, y=250)

button5 = tkinter.Button(win, text="FFT", command=FFT_)
button5.place(x=600, y=300)

button6 = tkinter.Button(win, text="Inverse FFT", command=IFFT_)
button6.place(x=600, y=350)

button7 = tkinter.Button(win, text="DCT", command=DCT_)
button7.place(x=600, y=400)

button8 = tkinter.Button(win, text="Inverse DCT", command=IDCT_)
button8.place(x=600, y=450)


button0 = tkinter.Button(win, text="Exit", command=win.quit)
button0.place(x=600, y=550)
win.mainloop()