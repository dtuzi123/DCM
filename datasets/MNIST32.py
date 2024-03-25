
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets.Data_Loading import *

class Generated_Dataset_Unsupervised(Dataset):
    def __init__(self ,filepath,datax,transform_train):
        #z = np.loadtxt(filepath ,dtype=np.float32 ,delimiter=',')

        self.transform = transform_train

        #mnist_train_x = mnist_train_x * 255.0
        #mnist_test = mnist_test * 255.0

        #mnist_train_x = np.transpose(mnist_train_x, (0, 3, 1, 2))
        #mnist_test = np.transpose(mnist_test, (0, 3, 1, 2))
        #mnist_train_x = mnist_train_x / 127.5 - 1
        #mnist_test = mnist_test / 127.5 - 1

        #self.x_data = torch.from_numpy(mnist_train_x)
        self.x_data = datax
        self.len = np.shape(self.x_data)[0]

    def GetSourceData(self):
        return self.x_data

    def SetData(self,datax):
        self.x_data = datax
        self.len = np.shape(self.x_data)[0]

    def GetLength(self):
        return self.len

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        xx = self.transform(self.x_data[item])
        xx = xx.type(torch.float)
        return xx


class Generated_Dataset(Dataset):
    def __init__(self ,filepath,datax,datay,transform_train):
        #z = np.loadtxt(filepath ,dtype=np.float32 ,delimiter=',')

        self.transform = transform_train

        #mnist_train_x = mnist_train_x * 255.0
        #mnist_test = mnist_test * 255.0

        #mnist_train_x = np.transpose(mnist_train_x, (0, 3, 1, 2))
        #mnist_test = np.transpose(mnist_test, (0, 3, 1, 2))
        #mnist_train_x = mnist_train_x / 127.5 - 1
        #mnist_test = mnist_test / 127.5 - 1

        #self.x_data = torch.from_numpy(mnist_train_x)
        self.x_data = datax
        mnist_train_label = np.argmax(datay,1)
        self.y_data = torch.from_numpy(mnist_train_label)
        self.len = self.x_data.shape[0]

    def SetData(self,datax,datay):
        self.x_data = datax
        mnist_train_label = np.argmax(datay, 1)
        self.y_data = torch.from_numpy(mnist_train_label)
        self.len = self.x_data.shape[0]

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        xx = self.transform(self.x_data[item])
        xx = xx.type(torch.float)
        return xx,self.y_data[item]

class MNIST32_Train(Dataset):
    def __init__(self ,filepath,transform_train):
        #z = np.loadtxt(filepath ,dtype=np.float32 ,delimiter=',')

        self.transform = transform_train
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        #mnist_train_x = mnist_train_x * 255.0
        #mnist_test = mnist_test * 255.0

        #mnist_train_x = np.transpose(mnist_train_x, (0, 3, 1, 2))
        #mnist_test = np.transpose(mnist_test, (0, 3, 1, 2))
        #mnist_train_x = mnist_train_x / 127.5 - 1
        #mnist_test = mnist_test / 127.5 - 1

        #self.x_data = torch.from_numpy(mnist_train_x)
        self.x_data = mnist_train_x
        mnist_train_label = np.argmax(mnist_train_label,1)
        self.y_data = torch.from_numpy(mnist_train_label)
        self.len = self.x_data.shape[0]

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        xx = self.transform(self.x_data[item])
        xx = xx.type(torch.float)
        return xx,self.y_data[item]


class MNIST32_Test(Dataset):
    def __init__(self ,filepath,transform_train):
        #z = np.loadtxt(filepath ,dtype=np.float32 ,delimiter=',')

        self.transform = transform_train
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        #mnist_train_x = mnist_train_x * 255.0
        #mnist_test = mnist_test * 255.0

        #mnist_train_x = np.transpose(mnist_train_x, (0, 3, 1, 2))
        #mnist_test = np.transpose(mnist_test, (0, 3, 1, 2))
        #mnist_train_x = mnist_train_x / 127.5 - 1
        #mnist_test = mnist_test / 127.5 - 1

        #self.x_data = torch.from_numpy(mnist_test)
        self.x_data = mnist_test
        mnist_label_test = np.argmax(mnist_label_test,1)
        self.y_data = torch.from_numpy(mnist_label_test)
        self.len = self.x_data.shape[0]

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        xx = self.transform(self.x_data[item])
        xx = xx.type(torch.float)
        return xx ,self.y_data[item]