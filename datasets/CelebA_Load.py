from torch.utils.data import Dataset #Dataset的包
import os #路径需要这个
import cv2 # 需要读取图片，最好用opencv-python,当然也可以用PIL只是我不顺手
import random
import numpy as np
import os
import gzip
import cv2
import scipy.io as sio
from cv2_imageProcess import *
import glob
import imageio
import scipy
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision

class CelebA_dataset(Dataset): #我定义的这个类
    def __init__(self, root_dir, transform=None):
     #下面需要使用的变量，在__init__定义好，
        self.transform = transform

        img_path = glob.glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        celebaFiles = data_files
        maxSize = 50000
        maxTestingSize = 5000

        # maxSize = 100
        # maxTestingSize = 100

        # maxSize = 128
        # maxTestingSize = 50

        celebaTraining = celebaFiles[0:maxSize]
        self.img_path = celebaTraining #得到整体图片的路径（可取其中的一张一张的图像的名字）

    def __getitem__(self, idx):
    # 改写__getitem__(self,item)函数，最后得到图像，标签
    		#获取具体的一幅图像的名字
        img_name = self.img_path[idx]
        #获取一幅图像的详细地址
        img_item_path = img_name
        #用opencv来读取图像

        '''
        img = cv2.imread(img_item_path)
        #获取标签（这里简单写了aligned与original）
        label = self.label_dir
        return img, label
        '''

        '''
        celebaTrainSet = [GetImage_cv(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in img_item_path]
        '''

        results = GetImage_cv(
            img_name,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)

        results = np.transpose(results, (2, 0, 1))
        #celebaTrainSet = np.array(celebaTrainSet)
        #celebaTrainSet = np.transpose(celebaTrainSet, (0, 3, 1, 2))

        return results

    def __len__(self):
    #改写整体图像的大小
        return len(self.img_path)


def file_name2(file_dir):
    t1 = []
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "../rendered_chairs/" + a1 + "/renders/*.png"
            img_path = glob.glob(b1)
            t1.append(img_path)

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc


class Chair_dataset(Dataset): #我定义的这个类
    def __init__(self, root_dir, transform=None):
     #下面需要使用的变量，在__init__定义好，
        self.transform = transform

        file_dir = "../rendered_chairs/"
        files = file_name2(file_dir)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        chairFiles = data_files

        maxSize = 50000
        maxTestingSize = 5000

        # maxSize = 100
        # maxTestingSize = 100

        # maxSize = 128
        # maxTestingSize = 50

        chairTraining = chairFiles[0:maxSize]

        #img_path = glob.glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片

        self.img_path = chairTraining #得到整体图片的路径（可取其中的一张一张的图像的名字）

    def __getitem__(self, idx):
    # 改写__getitem__(self,item)函数，最后得到图像，标签
    		#获取具体的一幅图像的名字
        img_name = self.img_path[idx]
        #获取一幅图像的详细地址
        img_item_path = img_name
        #用opencv来读取图像

        '''
        img = cv2.imread(img_item_path)
        #获取标签（这里简单写了aligned与original）
        label = self.label_dir
        return img, label
        '''

        '''
        celebaTrainSet = [GetImage_cv(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in img_item_path]
        '''

        results = get_image2(img_item_path, 300, is_crop=True, resize_w=image_size, is_grayscale=0)

        results = np.transpose(results, (2, 0, 1))
        #celebaTrainSet = np.array(celebaTrainSet)
        #celebaTrainSet = np.transpose(celebaTrainSet, (0, 3, 1, 2))

        return results

    def __len__(self):
    #改写整体图像的大小
        return len(self.img_path)
