import cv2
import numpy as np
import skimage.io as io
from skimage import io, transform

def center_crop_cv(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h

    h, w = x.shape[:2]

    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))

    tt = x[j:j + crop_h, i:i + crop_w]
    tt = np.array(tt)
    return cv2.resize(src=tt, dsize=(resize_h, resize_w))

def ims_cv(file,image):
    image = (image + 1) * 127.5
    #cv2.imwrite(file, image)
    io.imsave(file, image)

def ims_cv_255(file,image):
    image = image * 255.0
    #cv2.imwrite(file, image)
    io.imsave(file, image)

def center_crop2(x, crop_h, crop_w=None, resize_w=64):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]

    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(src=x[j:j+crop_h, i:i+crop_w], dsize=(resize_w, resize_w))


def GetImage_cv2_255(file, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    im = cv2.imread(file,cv2.IMREAD_COLOR)

    if is_crop == True:
        im = center_crop2(im,image_size, resize_w=resize_w)
    else:
        im = cv2.resize(src=im, dsize=(resize_w, resize_w))
    # im = cv2.resize(src=im, dsize=(64,64), interpolation=cv2.INTER_LINEAR)
    im = im / 255.0
    return im

def GetImage_cv2(file, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    #im = cv2.imread(file,cv2)
    im = io.imread(file)

    if is_crop == True:
        im = center_crop2(im,image_size, resize_w=resize_w)
    else:
        im = cv2.resize(src=im, dsize=(resize_w, resize_w))
    # im = cv2.resize(src=im, dsize=(64,64), interpolation=cv2.INTER_LINEAR)
    im = im / 127.5 -1
    return im



def GetImage_cv_01_Low(file,input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True):
    im = io.imread(file,False)
    #im = cv2.imread(file,cv2.IMREAD_COLOR)

    if crop == True:
        im = center_crop_cv(im,input_height, input_width, resize_height,resize_width)
    else:
        im = cv2.resize(src=im, dsize=(resize_height, resize_width))
    # im = cv2.resize(src=im, dsize=(64,64), interpolation=cv2.INTER_LINEAR)
    im = im / 127.5 -1
    return im


def GetImage_cv_255_Low(file,input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True):
    im = io.imread(file,False)
    #im = cv2.imread(file,cv2.IMREAD_COLOR)

    if crop == True:
        im = center_crop_cv(im,input_height, input_width, resize_height,resize_width)
    else:
        im = cv2.resize(src=im, dsize=(resize_height, resize_width))
    # im = cv2.resize(src=im, dsize=(64,64), interpolation=cv2.INTER_LINEAR)
    im = im / 255.0
    return im

import imageio

def GetImage_cv_255_Low_Specific(file,input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True):
    #im = io.imread(file,False)
    im = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    print(file)
    #im = imageio.imread(file)

    '''
    if crop == True:
        im = center_crop_cv(im,input_height, input_width, resize_height,resize_width)
    else:
        im = cv2.resize(src=im, dsize=(resize_height, resize_width))
    # im = cv2.resize(src=im, dsize=(64,64), interpolation=cv2.INTER_LINEAR)
    '''
    im = im / 255.0
    return im


def GetImage_cv_255(file,input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True):
    im = io.imread(file)
    #im = cv2.imread(file,cv2.IMREAD_COLOR)

    if crop == True:
        im = center_crop_cv(im,input_height, input_width, resize_height,resize_width)
    else:
        im = cv2.resize(src=im, dsize=(resize_height, resize_width))
    # im = cv2.resize(src=im, dsize=(64,64), interpolation=cv2.INTER_LINEAR)
    im = im / 255.0
    return im

def GetImage_cv(file,input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True):
    #im = cv2.imread(file)
    im = io.imread(file)

    if crop == True:
        im = center_crop_cv(im,input_height, input_width, resize_height,resize_width)
    else:
        im = cv2.resize(src=im, dsize=(resize_height, resize_width))
    # im = cv2.resize(src=im, dsize=(64,64), interpolation=cv2.INTER_LINEAR)
    im = im / 127.5 -1

    #im = im[:, :, ::-1]
    return im
