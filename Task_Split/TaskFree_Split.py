import numpy as np

from Task_Split.Task_utilizes import *
from datasets.Data_Loading import *
from datasets.Fid_evaluation import *
import torch
from datasets.MyCIFAR10 import *


def Give_DataStream_CIFAR100_Unsupervised():
    file = "data/cifar-100-python/train"
    train_data = unpickle(file)

    file2 = "data/cifar-100-python/test"
    test_data = unpickle(file2)

    X_train = train_data['data']
    Y_train = train_data['fine_labels']

    X_test = test_data['data']
    Y_test = test_data['fine_labels']

    X_train = X_train / 127.5 - 1
    X_test = X_test / 127.5 - 1

    X_train = np.reshape(X_train, (-1, 32, 32, 3))
    X_test = np.reshape(X_test, (-1, 32, 32, 3))
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    Y_train = np.argmax(Y_train,1)
    Y_test = np.argmax(Y_test,1)
    X_train, Y_train = Split_DataSet_CIFAR100(X_train,Y_train)
    X_test, Y_test = Split_DataSet_CIFAR100(X_test,Y_test)
    return X_train, Y_train,X_test, Y_test


def Give_DataStream_CIFAR100_Supervised():
    file = "data/cifar-100-python/train"
    train_data = unpickle(file)

    file2 = "data/cifar-100-python/test"
    test_data = unpickle(file2)

    for item in train_data:
        print(item, type(train_data[item]))

    X_train = train_data[b'data']
    Y_train = train_data[b'fine_labels']

    X_test = test_data[b'data']
    Y_test = test_data[b'fine_labels']

    X_train = X_train / 127.5 - 1
    X_test = X_test / 127.5 - 1

    X_train = np.reshape(X_train, (-1, 32, 32, 3))
    X_test = np.reshape(X_test, (-1, 32, 32, 3))
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    #Y_train = np.argmax(Y_train,1)
    #Y_test = np.argmax(Y_test,1)

    X_train, Y_train = Split_DataSet_CIFAR100(X_train,Y_train)
    X_test, Y_test = Split_DataSet_CIFAR100(X_test,Y_test)
    return X_train, Y_train,X_test, Y_test


def Give_DataStream_Unsupervised_Numpy(dataSet):
    if dataSet == "mnist":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        #train_data = np.transpose(train_data, (0, 3, 1, 2))
        #test_data = np.transpose(test_data, (0, 3, 1, 2))
        #train_data = train_data / 127.5 - 1
        #test_data = test_data / 127.5 - 1
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)


    elif dataSet == "fashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        #train_data = np.transpose(train_data, (0, 3, 1, 2))
        #test_data = np.transpose(test_data, (0, 3, 1, 2))
        #train_data = train_data / 127.5 - 1
        #test_data = test_data / 127.5 - 1
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)


    elif dataSet == "svhn":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()
        # FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = x_train
        test_data = x_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = y_train
        test_labels = y_test

        #train_data = np.transpose(train_data, (0, 3, 1, 2))
        #test_data = np.transpose(test_data, (0, 3, 1, 2))
        #train_data = train_data / 127.5 - 1
        #test_data = test_data / 127.5 - 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        train_labels = np.reshape(train_labels, (-1, 10))
        test_labels = np.reshape(test_labels, (-1, 10))

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)


    elif dataSet == "cifar10":
        train_data, train_labels, test_data, test_labels = prepare_data()

        #train_data = np.transpose(train_data, (0, 3, 1, 2))
        #test_data = np.transpose(test_data, (0, 3, 1, 2))
        #train_data = train_data / 127.5 - 1
        #test_data = test_data / 127.5 - 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)

    dataStramX = []
    dataStramY = []
    for i in range(np.shape(totalSetX)[0]):
        if np.shape(dataStramX)[0] == 0:
            dataStramX = totalSetX[i]
            dataStramY = totalSetY[i]
        else:
            dataStramX = np.concatenate((dataStramX, totalSetX[i]), axis=0)
            dataStramY = np.concatenate((dataStramY, totalSetY[i]), axis=0)

    dataStramX = np.array(dataStramX)
    dataStramY = np.array(dataStramY)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dataStramX = torch.tensor(dataStramX).cuda().to(device=device, dtype=torch.float)

    return dataStramX,totalTestX


def rotate_img(img, angle):
    '''
    img   --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    #获取旋转矩阵
    # 参数1为旋转中心点;
    # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
    # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    #计算图像新边界
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    #调整旋转矩阵以考虑平移
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img

def Dataset_Rotate(dataset,angle):
    arr = []
    count = int(np.shape(dataset)[0])
    for i in range(count):
        a1 = rotate_img(dataset[i],angle)
        arr.append(a1)

    arr = np.array(arr)
    return arr


def InverseDataset(dataset):
    dataX = dataset
    for i in range(np.shape(dataset)[0]):
        for k1 in range(3):
            for k2 in range(32):
                for k3 in range(32):
                    dataX[i, k1, k2,k3] = 255.0 - dataX[i, k1, k2,k3]
    return dataX

def GiveDataStream_SixTask_NonSplit():
    MnistStramX, totalTestX, MnistTest = Give_Dataset_Unsupervised("mnist")
    SvhnStramX, totalTestX, SvhnTest = Give_Dataset_Unsupervised("svhn")
    FashionStramX, totalTestX, FashionTest = Give_Dataset_Unsupervised("fashion")
    InverseFashionStramX, totalTestX, InverseFashionTest = Give_Dataset_Unsupervised("InverseFashion")
    RotateMNISTStramX, totalTestX, RotateMNISTTest = Give_Dataset_Unsupervised("RotateMNIST")
    Cifar10StramX, totalTestX, Cifar10Test = Give_Dataset_Unsupervised("cifar10")

    dataS = np.concatenate((MnistStramX,SvhnStramX),0)
    dataS = np.concatenate((dataS,FashionStramX),0)
    dataS = np.concatenate((dataS,InverseFashionStramX),0)
    dataS = np.concatenate((dataS,RotateMNISTStramX),0)
    dataS = np.concatenate((dataS,Cifar10StramX),0)

    return dataS,MnistTest,SvhnTest,FashionTest,InverseFashionTest,RotateMNISTTest,Cifar10Test


def GiveDataStream_SixTask_Split():
    MnistStramX, totalTestX, MnistTest = Give_DataStream_Unsupervised_Numpy("mnist")
    SvhnStramX, totalTestX, SvhnTest = Give_DataStream_Unsupervised_Numpy("svhn")
    FashionStramX, totalTestX, FashionTest = Give_DataStream_Unsupervised_Numpy("fashion")
    InverseFashionStramX, totalTestX, InverseFashionTest = Give_DataStream_Unsupervised_Numpy("InverseFashion")
    RotateMNISTStramX, totalTestX, RotateMNISTTest = Give_DataStream_Unsupervised_Numpy("RotateMNIST")
    Cifar10StramX, totalTestX, Cifar10Test = Give_DataStream_Unsupervised_Numpy("cifar10")

    dataS = np.concatenate((MnistStramX,SvhnStramX),0)
    dataS = np.concatenate((dataS,FashionStramX),0)
    dataS = np.concatenate((dataS,InverseFashionStramX),0)
    dataS = np.concatenate((dataS,RotateMNISTStramX),0)
    dataS = np.concatenate((dataS,Cifar10StramX),0)

    return dataS,MnistTest,SvhnTest,FashionTest,InverseFashionTest,RotateMNISTTest,Cifar10Test


def Give_Dataset_Unsupervised(dataSet):

    defaultTest = 0
    if dataSet == "mnist":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        #totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
        totalSetX = train_data

        totalTestX = test_data
        totalTestY = test_labels

    elif dataSet == "fashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        #totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)

        totalSetX = train_data

        totalTestX = test_data
        totalTestY = test_labels

    elif dataSet == "svhn":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()
        # FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = x_train
        test_data = x_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = y_train
        test_labels = y_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        train_labels = np.reshape(train_labels, (-1, 10))
        test_labels = np.reshape(test_labels, (-1, 10))

        #totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
        #                                                                 test_labels)

        totalSetX = train_data

        totalTestX = test_data
        totalTestY = test_labels


    elif dataSet == "cifar10":

        train_data, train_labels, test_data, test_labels = prepare_data()

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        defaultTest = test_data

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        #totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
        #                                                                  test_labels)

        totalSetX = train_data

        totalTestX = test_data
        totalTestY = test_labels
    elif dataSet == "RotateMNIST":

        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        # totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
        totalSetX = train_data

        totalTestX = test_data
        totalTestY = test_labels

        defaultTest = Dataset_Rotate(defaultTest,180)
        totalSetX = Dataset_Rotate(totalSetX,180)


    elif dataSet == "InverseFashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        train_data = InverseDataset(train_data)
        test_data = InverseDataset(test_data)

        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        # totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)

        totalSetX = train_data

        totalTestX = test_data
        totalTestY = test_labels


    dataStramX = []
    dataStramY = []
    dataStramX = totalSetX

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dataStramX = torch.tensor(dataStramX).cuda().to(device=device, dtype=torch.float)
    #totalTestX = torch.tensor(totalTestX).cuda().to(device=device, dtype=torch.float)

    return dataStramX,totalTestX,defaultTest

def Give_DataStream_Unsupervised(dataSet):

    defaultTest = 0
    if dataSet == "mnist":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        totalTestX = test_data
        totalTestY = test_labels

    elif dataSet == "fashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)

        totalTestX = test_data
        totalTestY = test_labels

    elif dataSet == "svhn":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()
        # FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = x_train
        test_data = x_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = y_train
        test_labels = y_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        train_labels = np.reshape(train_labels, (-1, 10))
        test_labels = np.reshape(test_labels, (-1, 10))

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        totalTestX = test_data
        totalTestY = test_labels


    elif dataSet == "cifar10":
        train_data, train_labels, test_data, test_labels = prepare_data()

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        defaultTest = test_data

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        totalTestX = test_data
        totalTestY = test_labels

    elif dataSet == "RotateMNIST":

        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        # totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
        totalSetX = train_data

        totalTestX = test_data
        totalTestY = test_labels

        train_data = Dataset_Rotate(train_data,180)
        test_data = Dataset_Rotate(test_data,180)

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)


        totalTestX = test_data
        totalTestY = test_labels


    elif dataSet == "InverseFashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        train_data = InverseDataset(train_data)
        test_data = InverseDataset(test_data)

        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        # totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)

        totalSetX = train_data

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)

        totalTestX = test_data
        totalTestY = test_labels


    dataStramX = []
    dataStramY = []
    for i in range(np.shape(totalSetX)[0]):
        if np.shape(dataStramX)[0] == 0:
            dataStramX = totalSetX[i]
            dataStramY = totalSetY[i]
        else:
            dataStramX = np.concatenate((dataStramX, totalSetX[i]), axis=0)
            dataStramY = np.concatenate((dataStramY, totalSetY[i]), axis=0)

    dataStramX = np.array(dataStramX)
    dataStramY = np.array(dataStramY)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataStramX = torch.tensor(dataStramX).cuda().to(device=device, dtype=torch.float)

    return dataStramX,totalTestX,defaultTest

def Give_DataStream_Unsupervised_Numpy(dataSet):

    defaultTest = 0
    if dataSet == "mnist":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        totalTestX = test_data
        totalTestY = test_labels

    elif dataSet == "fashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)

        totalTestX = test_data
        totalTestY = test_labels

    elif dataSet == "svhn":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()
        # FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = x_train
        test_data = x_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = y_train
        test_labels = y_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        train_labels = np.reshape(train_labels, (-1, 10))
        test_labels = np.reshape(test_labels, (-1, 10))

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        totalTestX = test_data
        totalTestY = test_labels


    elif dataSet == "cifar10":
        train_data, train_labels, test_data, test_labels = prepare_data()

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        defaultTest = test_data

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        totalTestX = test_data
        totalTestY = test_labels

    elif dataSet == "RotateMNIST":

        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        # totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
        totalSetX = train_data

        totalTestX = test_data
        totalTestY = test_labels

        train_data = Dataset_Rotate(train_data,180)
        test_data = Dataset_Rotate(test_data,180)

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)


        totalTestX = test_data
        totalTestY = test_labels


    elif dataSet == "InverseFashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        train_data = InverseDataset(train_data)
        test_data = InverseDataset(test_data)

        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1

        defaultTest = test_data
        # totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)

        totalSetX = train_data

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)

        totalTestX = test_data
        totalTestY = test_labels


    dataStramX = []
    dataStramY = []
    for i in range(np.shape(totalSetX)[0]):
        if np.shape(dataStramX)[0] == 0:
            dataStramX = totalSetX[i]
            dataStramY = totalSetY[i]
        else:
            dataStramX = np.concatenate((dataStramX, totalSetX[i]), axis=0)
            dataStramY = np.concatenate((dataStramY, totalSetY[i]), axis=0)

    dataStramX = np.array(dataStramX)
    dataStramY = np.array(dataStramY)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataStramX,totalTestX,defaultTest


def CombineDataset(dataX):
    result = []
    count = np.shape(dataX)[0]
    for i in range(count):
        if np.shape(result)[0] == 0:
            result = dataX[i]
        else:
            result = np.concatenate((result,dataX[i]),0)
    return result

def CombineDataset_WithRandom(dataX):
    result = []
    count = np.shape(dataX)[0]
    for i in range(count):
        if np.shape(result)[0] == 0:
            result = dataX[i]
        else:
            result = np.concatenate((result,dataX[i]),0)

    n_examples = np.shape(result)[0]
    index2 = [i for i in range(n_examples)]
    np.random.shuffle(index2)
    result = result[index2]

    return result


def Give_DataStream_Supervised_2(dataSet):
    if dataSet == "mnist":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()


        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        defaultTest = totalTestX


    elif dataSet == "fashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)

        defaultTest = totalTestX

    elif dataSet == "svhn":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()
        # FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = x_train
        test_data = x_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = y_train
        test_labels = y_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        train_labels = np.reshape(train_labels, (-1, 10))
        test_labels = np.reshape(test_labels, (-1, 10))

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)

        defaultTest = totalTestX

    elif dataSet == "cifar10":
        train_data, train_labels, test_data, test_labels = prepare_data()

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        defaultTest = totalTestX

    dataStramX = []
    dataStramY = []
    for i in range(np.shape(totalSetX)[0]):
        if np.shape(dataStramX)[0] == 0:
            dataStramX = totalSetX[i]
            dataStramY = totalSetY[i]
        else:
            dataStramX = np.concatenate((dataStramX, totalSetX[i]), axis=0)
            dataStramY = np.concatenate((dataStramY, totalSetY[i]), axis=0)

    dataStramX = np.array(dataStramX)
    dataStramY = np.array(dataStramY)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataStramX = torch.tensor(dataStramX).cuda().to(device=device, dtype=torch.float)
    dataStramY = torch.tensor(dataStramY).cuda().to(device=device)

    return dataStramX,dataStramY,defaultTest,totalTestY


def GiveDataStream_SixTask(isNumpy):

    #Data stream
    dataS = []

    #mnist
    mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

    train_data = mnist_train_x
    test_data = mnist_test
    train_data = train_data * 255.0
    test_data = test_data * 255.0
    train_labels = mnist_train_label
    test_labels = mnist_label_test

    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))
    train_data = train_data / 127.5 - 1
    test_data = test_data / 127.5 - 1

    mnistTest = test_data

    dataS = train_data

    #SVHN
    svhnTrainX = x_train
    svhnTestX = x_test
    svhnTrainX = svhnTrainX * 255.0
    svhnTestX = svhnTestX * 255.0
    svhnTrainX = np.transpose(svhnTrainX, (0, 3, 1, 2))
    svhnTestX = np.transpose(svhnTestX, (0, 3, 1, 2))
    svhnTrainX = svhnTrainX / 127.5 - 1
    svhnTestX = svhnTestX / 127.5 - 1

    dataS = np.concatenate((dataS,svhnTrainX),0)

    #Fashion
    FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

    FashionTrain_x = FashionTrain_x*255.0
    FashionTest_x = FashionTest_x*255.0
    FashionTrain_x = np.transpose(FashionTrain_x, (0, 3, 1, 2))
    FashionTest_x = np.transpose(FashionTest_x, (0, 3, 1, 2))
    FashionTrain_x = FashionTrain_x / 127.5 - 1
    FashionTest_x = FashionTest_x / 127.5 - 1

    dataS = np.concatenate((dataS,FashionTrain_x),0)

    #InverseFashion
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X, (-1, 28, 28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i, k1, k2] = 1.0 - data_X[i, k1, k2]

    data_X = np.reshape(data_X, (-1, 28, 28, 1))
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)
    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    myArr = np.array(myArr)
    data_X = myArr

    InverseFashionTrain_x = data_X[0:60000]
    InverseFashionTest_x = data_X[60000:70000]

    InverseFashionTrain_x = InverseFashionTrain_x * 255.0
    InverseFashionTest_x = InverseFashionTest_x * 255.0
    InverseFashionTrain_x = np.transpose(InverseFashionTrain_x, (0, 3, 1, 2))
    InverseFashionTest_x = np.transpose(InverseFashionTest_x, (0, 3, 1, 2))
    InverseFashionTrain_x = InverseFashionTrain_x / 127.5 - 1
    InverseFashionTest_x = InverseFashionTest_x / 127.5 - 1

    dataS = np.concatenate((dataS,InverseFashionTrain_x),0)

    #RMNIST
    mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()


    #rote operation
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    Rmnist_train_x = []
    for i in range(np.shape(mnist_train_x)[0]):
        x = mnist_train_x[i]
        rotated_im = rot_img(x, np.pi, dtype)  # Rotate image by 90 degrees.
        if np.shape(Rmnist_train_x)[0] == 0:
            Rmnist_train_x = rotated_im
        else:
            Rmnist_train_x = torch.cat([Rmnist_train_x,rotated_im],0)

    Rmnist_test = []
    for i in range(np.shape(mnist_test)[0]):
        x = mnist_train_x[i]
        rotated_im = rot_img(x, np.pi, dtype)  # Rotate image by 90 degrees.
        if np.shape(Rmnist_test)[0] == 0:
            Rmnist_train_x = rotated_im
        else:
            Rmnist_train_x = torch.cat([Rmnist_train_x, rotated_im], 0)

    Rmnist_train_x = Transfer_To_Numpy(Rmnist_train_x)
    Rmnist_test = Transfer_To_Numpy(Rmnist_test)

    Rmnist_train_x = mnist_train_x*255.0
    Rmnist_test = mnist_test * 255.0

    Rmnist_train_x = np.transpose(Rmnist_train_x, (0, 3, 1, 2))
    Rmnist_test = np.transpose(Rmnist_test, (0, 3, 1, 2))
    Rmnist_train_x = Rmnist_train_x / 127.5 - 1
    Rmnist_test = Rmnist_test / 127.5 - 1

    dataS = np.concatenate((dataS,Rmnist_train_x),0)

    #cifar10
    train_data, train_labels, test_data, test_labels = prepare_data()

    cifarTrainX = train_data
    cifarTestX = test_data

    cifarTrainX = np.transpose(cifarTrainX, (0, 3, 1, 2))
    cifarTestX = np.transpose(cifarTestX, (0, 3, 1, 2))
    cifarTrainX = cifarTrainX / 127.5 - 1
    cifarTestX = cifarTestX / 127.5 - 1

    dataS = np.concatenate((dataS,cifarTrainX),0)

    return dataS,mnistTest,svhnTestX,FashionTest_x,InverseFashionTest_x,Rmnist_test,cifarTestX

def Transfer_To_Numpy(sample):
    mySamples = sample.unsqueeze(0).cuda().cpu()
    mySamples = np.array(mySamples)
    mySamples = mySamples[0]
    return mySamples

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

def Give_DataStream_Supervised(dataSet):
    if dataSet == "mnist":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        train_data = mnist_train_x
        test_data = mnist_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = mnist_train_label
        test_labels = mnist_label_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        defaultTest = test_data


    elif dataSet == "fashion":
        FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = FashionTrain_x
        test_data = FashionTest_x
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = FashionTrain_label
        test_labels = FashionTest_label

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data,train_labels,test_data,test_labels)

        defaultTest = test_data

    elif dataSet == "svhn":
        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()
        # FashionTrain_x, FashionTrain_label, FashionTest_x, FashionTest_label = GiveFashion32()

        train_data = x_train
        test_data = x_test
        train_data = train_data * 255.0
        test_data = test_data * 255.0
        train_labels = y_train
        test_labels = y_test

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        train_labels = np.reshape(train_labels, (-1, 10))
        test_labels = np.reshape(test_labels, (-1, 10))

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)

        defaultTest = test_data

    elif dataSet == "cifar10":
        train_data, train_labels, test_data, test_labels = prepare_data()

        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        train_data = train_data / 127.5 - 1
        test_data = test_data / 127.5 - 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # train_data = torch.tensor(train_data).cuda().to(device=device, dtype=torch.float)
        # test_data = torch.tensor(test_data).cuda().to(device=device, dtype=torch.float)

        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        defaultTest = test_data

    elif dataSet == "inversefashion":
        train_data, train_labels, test_data, test_labels = Give_InverseFashion32()
        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))
        totalSetX, totalSetY, totalTestX, totalTestY = Split_Dataset_Five(train_data, train_labels, test_data,
                                                                          test_labels)
        defaultTest = test_data


    dataStramX = []
    dataStramY = []
    for i in range(np.shape(totalSetX)[0]):
        if np.shape(dataStramX)[0] == 0:
            dataStramX = totalSetX[i]
            dataStramY = totalSetY[i]
        else:
            dataStramX = np.concatenate((dataStramX, totalSetX[i]), axis=0)
            dataStramY = np.concatenate((dataStramY, totalSetY[i]), axis=0)

    dataStramX = np.array(dataStramX)
    dataStramY = np.array(dataStramY)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataStramX = torch.tensor(dataStramX).cuda().to(device=device, dtype=torch.float)
    dataStramY = torch.tensor(dataStramY).cuda().to(device=device)

    #defaultTest = np.transpose(defaultTest, (0, 3, 1, 2))

    return dataStramX,dataStramY,defaultTest,test_labels


