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

import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    #return Image.fromarray(x[j:j+crop_h, i:i+crop_w]).resize([resize_w, resize_w])
    xx = x[j:j+crop_h, i:i+crop_w]
    xx = xx.astype(np.uint8)

    return Image.fromarray(xx).resize(size = (resize_w, resize_w))
    #return scipy.misc.imresize(xx,
    #                           [resize_w, resize_w])
    '''
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])
    '''

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.  # change pixel value range from [0,255] to [-1,1] to feed into CNN

def inverse_transform(images):
    return (images+1.)/2. # change image pixel value(outputs from tanh in range [-1,1]) back to [0,1]

def imread(path, is_grayscale = False):
    '''
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float) # [width,height] flatten RGB image to grayscale image
    else:
        return scipy.misc.imread(path).astype(np.float) # [width,height,color_dim]
    '''
    if (is_grayscale):
        return imageio.imread(path, flatten = True).astype(np.float) # [width,height] flatten RGB image to grayscale image
    else:
        return imageio.imread(path).astype(np.float) # [width,height,color_dim]


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def get_image2(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

#load large datasets

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

def Load_SplitTinyImageNet():
    data_dir = '../tiny-imagenet-200/'
    num_workers = {'train': 100, 'val': 0, 'test': 0}
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=False, num_workers=num_workers[x])
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    trainLoad = image_datasets['train']
    testLoad = image_datasets['test']

    totalArr = []
    arr = []

    currentIndex = 0
    b = 0
    for i, (inputs, labels) in enumerate(trainLoad):
        if labels >= currentIndex and labels < currentIndex + 5:
            #inputs = inputs.view(1,3, 64,64)
            inputs = inputs.unsqueeze(0).cuda().cpu()
            inputs = np.reshape(inputs,(1,3,64,64))
            if np.shape(arr)[0] == 0:
                arr = inputs
            else:
                #arr = torch.cat([arr, inputs], 0)
                arr = np.concatenate((arr,inputs),0)
        else:
            b = b+1
            n_examples = np.shape(arr)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            arr = arr[index2]

            if np.shape(totalArr)[0] == 0:
                totalArr = arr
            else:
                totalArr = np.concatenate((totalArr,arr),0)

    testData = []

    for i, (inputs, labels) in enumerate(testLoad):
        inputs = inputs.unsqueeze(0).cuda().cpu()
        inputs = np.reshape(inputs, (1, 3, 64, 64))
        if np.shape(testData)[0] == 0:
            testData = inputs
        else:
            testData = np.concatenate((testData,inputs),0)

    dataStream = totalArr
    print(np.shape(dataStream))
    return dataStream,testData

def Load_TinyImageNet():
    img_path = glob.glob('../train_64x64/*.png')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    imageNetFiles = data_files  # [0:2000]
    print("aaa")
    print(np.shape(imageNetFiles))

    imageNetTrain = imageNetFiles[0:50000]
    imageNetFilesTest = imageNetFiles[50000:55000]

    imageNetTrain = [GetImage_cv(
        sample_file,
        input_height=64,
        input_width=64,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in imageNetTrain]

    imageNetTestData = [GetImage_cv(
        sample_file,
        input_height=64,
        input_width=64,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in imageNetFilesTest]

    imageNetTrain = np.array(imageNetTrain)
    imageNetTestData = np.array(imageNetTestData)
    return imageNetTrain, imageNetTestData



def Load_ImageNet():
    img_path = glob.glob('../train_64x64/*.png')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    imageNetFiles = data_files  # [0:2000]
    print("aaa")
    print(np.shape(imageNetFiles))

    imageNetTrain = imageNetFiles[0:50000]
    imageNetFilesTest = imageNetFiles[50000:55000]

    imageNetTrain = [GetImage_cv(
        sample_file,
        input_height=64,
        input_width=64,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in imageNetTrain]

    imageNetTestData = [GetImage_cv(
        sample_file,
        input_height=64,
        input_width=64,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in imageNetFilesTest]

    imageNetTrain = np.transpose(imageNetTrain, (0, 3, 1, 2))
    imageNetTestData = np.transpose(imageNetTestData, (0, 3, 1, 2))

    imageNetTrain = np.array(imageNetTrain)
    imageNetTestData = np.array(imageNetTestData)
    return imageNetTrain,imageNetTestData


def CombineData(totalSet, labelSet, startIndex, endIndex):
    arr1 = []
    totalLabel = []
    for i in range(startIndex, endIndex + 1):
        print(i)
        set1 = totalSet[i]
        label1 = labelSet[i]
        label1 = np.copy(label1)
        for j in range(np.shape(set1)[0]):
            arr1.append(set1[j])
            totalLabel.append(label1)

    arr1 = np.array(arr1)
    totalLabel = np.array(totalLabel)

    n_examples = np.shape(arr1)[0]
    index2 = [i for i in range(n_examples)]
    np.random.shuffle(index2)
    arr1 = arr1[index2]
    totalLabel = totalLabel[index2]

    return arr1, totalLabel


def SplitMNISTImage_SingleTask(totalSet, labelSet):
    totalArr = []
    totalArrY = []

    arrY = []
    arr = []
    currentIndex = 0
    for i in range(np.shape(totalSet)[0]):
        batch = totalSet[i]
        batchY = labelSet[i]
        if batchY > currentIndex:
            currentIndex = batchY
            totalArr.append(arr)
            totalArrY.append(arrY)
            arr = []
            arrY = []
        else:
            arr.append(batch)
            arrY.append(batchY)

def CreateDataset(set,label):
    totalLabel = np.zeros((np.shape(set)[0],100))
    totalLabel[:,label] = 1

    return set,totalLabel

def Split_MINI_ImageNet2_Final_2(totalSet, labelSet):
    n_task = 20

    totalTraining = []
    totalTrainingY = []
    totalTesting = []
    totalTestingY = []

    for i in range(np.shape(totalSet)[0]):
        training = totalSet[0:400]
        testing = totalSet[400:500]

        currentLabel = labelSet[i]
        currentLabel = np.argmax(currentLabel, 1)
        training,trainingY = CreateDataset(training,currentLabel)
        testing,testingY = CreateDataset(testing,currentLabel)

        totalTraining.append(training)
        totalTrainingY.append(trainingY)
        totalTesting.append(testing)
        totalTestingY.append(testingY)


    resultTraining = []
    resultTrainingY = []
    resultTesting = []
    resultTestingY = []

    for j in range(n_task):
        startIndex = j * 5
        endIndex = (j + 1) * 5 - 1
        arr1, label1 = CombineData(totalTraining, totalTrainingY, startIndex, endIndex)

        traingSet = arr1
        trainLabelSet = label1
        for j in range(np.shape(traingSet)[0]):
            resultTraining.append(traingSet[j])
            resultTrainingY.append(trainLabelSet[j])

        arr1, label1 = CombineData(totalTesting,totalTestingY , startIndex, endIndex)

        traingSet = arr1
        trainLabelSet = label1
        for j in range(np.shape(traingSet)[0]):
            resultTesting.append(traingSet[j])
            resultTestingY.append(trainLabelSet[j])

    return resultTraining,resultTrainingY,resultTesting,resultTestingY


def Split_MINI_ImageNet2_Final(totalSet, labelSet):
    n_task = 20

    trainingCount = 5 * 500
    totalTraingSet = []
    totalTrainLabeSet = []

    totalTestingSet = []
    totalTestingLabeSet = []

    testArr1 = []
    testingLabel1 = []

    testingCount = 200

    for i in range(n_task):
        startIndex = i * 5
        endIndex = (i + 1) * 5 - 1
        arr1, label1 = CombineData(totalSet, labelSet, startIndex, endIndex)

        if i <4:
            testSet = arr1
            testLabelSet = label1
            for j in range(np.shape(testSet)[0]):
                totalTestingSet.append(testSet[j])
                totalTestingLabeSet.append(testLabelSet[j])
        else:
            traingSet = arr1
            trainLabelSet = label1
            for j in range(np.shape(traingSet)[0]):
                totalTraingSet.append(traingSet[j])
                totalTrainLabeSet.append(trainLabelSet[j])

    testingCount = 500
    #totalTestingSet = totalTraingSet[0:testingCount]
    #totalTestingLabeSet = totalTrainLabeSet[0:testingCount]

    return totalTraingSet, totalTrainLabeSet, totalTestingSet, totalTestingLabeSet

def LoadImages_ForMINIImageNet(totalSet):
    batch = [GetImage_cv(
        batch_file,
        input_height=84,
        input_width=84,
        resize_height=64,
        resize_width=64,
        crop=False) \
        for batch_file in totalSet]
    batch = np.array(batch)
    batch = np.transpose(batch, (0, 3, 1, 2))

    return batch

def LoadMINIImageNet():
    LabelArr = []
    dataArr = []

    index = 0
    t1 = []
    file_dir = "../mini_imagenet/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            bFileDir = file_dir + a1
            # b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            # b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(bFileDir):
                for a2 in dirs2:
                    newLabel = np.zeros(100)
                    newLabel[index] = 1
                    b2 = bFileDir + "/" + a2 + "/*.jpg"
                    # print(b2)
                    img_path = glob.glob(b2)
                    t1.append(img_path)

                    LabelArr.append(newLabel)
                    index = index + 1
    #return t1, LabelArr

    totalTraingSet, totalTrainLabeSet, totalTestingSet, totalTestingLabeSet = Split_MINI_ImageNet2_Final(t1, LabelArr)

    print(np.shape(totalTraingSet)[0])
    print(np.shape(totalTestingSet)[0])

    totalTraingSet = LoadImages_ForMINIImageNet(totalTraingSet)
    totalTestingSet = LoadImages_ForMINIImageNet(totalTestingSet)
    return totalTraingSet,totalTestingSet

def LoadMINIImageNet_Supervised():
    LabelArr = []
    dataArr = []

    index = 0
    t1 = []
    file_dir = "../mini_imagenet/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            bFileDir = file_dir + a1
            # b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            # b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(bFileDir):
                for a2 in dirs2:
                    newLabel = np.zeros(100)
                    newLabel[index] = 1
                    b2 = bFileDir + "/" + a2 + "/*.jpg"
                    # print(b2)
                    img_path = glob.glob(b2)
                    t1.append(img_path)

                    LabelArr.append(newLabel)
                    index = index + 1
    #return t1, LabelArr

    totalTraingSet, totalTrainLabeSet, totalTestingSet, totalTestingLabeSet = Split_MINI_ImageNet2_Final_2(t1, LabelArr)

    print(np.shape(totalTraingSet)[0])
    print(np.shape(totalTestingSet)[0])

    totalTraingSet = LoadImages_ForMINIImageNet(totalTraingSet)
    totalTestingSet = LoadImages_ForMINIImageNet(totalTestingSet)
    return totalTraingSet,totalTrainLabeSet,totalTestingSet,totalTestingLabeSet


def Load_CACD_Files():
    img_path = glob.glob('../CACD2000/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    cacdFiles = data_files
    trainingSet = cacdFiles[0:50000]
    testingSet = cacdFiles[50000:51000]
    return trainingSet,testingSet
    '''
    print(np.shape(data_files))
    cacdFiles = cacdFiles[0:128]

    batch = [GetImage_cv(
        sample_file,
        input_height=250,
        input_width=250,
        resize_height=256,
        resize_width=256,
        crop=False)
        for sample_file in cacdFiles]

    print(np.shape(batch))

    batch = np.transpose(batch, (0, 3, 1, 2))

    return batch,batch
    '''

def Load_CACD():
    img_path = glob.glob('../CACD2000/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    cacdFiles = data_files
    trainingSet = cacdFiles[0:50000]
    testingSet = cacdFiles[50000:51000]
    return trainingSet,testingSet
    '''
    print(np.shape(data_files))
    cacdFiles = cacdFiles[0:128]

    batch = [GetImage_cv(
        sample_file,
        input_height=250,
        input_width=250,
        resize_height=256,
        resize_width=256,
        crop=False)
        for sample_file in cacdFiles]

    print(np.shape(batch))

    batch = np.transpose(batch, (0, 3, 1, 2))

    return batch,batch
    '''

def Load_CACD_Images():
    img_path = glob.glob('../CACD2000/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    cacdFiles = data_files
    trainingSet = cacdFiles[0:50000]
    testingSet = cacdFiles[50000:51000]
    #return trainingSet,testingSet

    print(np.shape(data_files))
    cacdFiles = cacdFiles[0:128]

    batch = [GetImage_cv(
        sample_file,
        input_height=250,
        input_width=250,
        resize_height=64,
        resize_width=64,
        crop=False)
        for sample_file in trainingSet]

    batch = np.transpose(batch, (0, 3, 1, 2))

    batch2 = [GetImage_cv(
        sample_file,
        input_height=250,
        input_width=250,
        resize_height=64,
        resize_width=64,
        crop=False)
        for sample_file in testingSet]

    batch2 = np.transpose(batch2, (0, 3, 1, 2))

    return batch,batch2


def Load_FFHQ128():
    file_dir = "../FFHQ128/"
    t1 = []
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            bFileDir = file_dir + a1 + "/*.png"
            # b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            # b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            img_path = glob.glob(bFileDir)
            #print(img_path)
            for jj in range(np.shape(img_path)[0]):
                t1.append(img_path[jj])

    t1 = np.array(t1)
    trainingSet = t1[0:50000]
    testingSet = t1[50000:51000]
    return trainingSet,testingSet

def Load_FFHQ():
    img_path = glob.glob('../FFHQ/resized/*.jpg')  # 获取新文件夹下所有图片
    img_path = glob.glob('../resized/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    cacdFiles = data_files
    trainingSet = cacdFiles[0:50000]
    testingSet = cacdFiles[50000:51000]
    return trainingSet,testingSet
    '''
    print(np.shape(data_files))
    cacdFiles = cacdFiles[0:128]

    batch = [GetImage_cv(
        sample_file,
        input_height=250,
        input_width=250,
        resize_height=256,
        resize_width=256,
        crop=False)
        for sample_file in cacdFiles]

    print(np.shape(batch))

    batch = np.transpose(batch, (0, 3, 1, 2))

    return batch,batch
    '''


def Load_CelebATest():
    img_path = glob.glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
    # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    celebaFiles = data_files
    maxSize = 100
    maxTestingSize = 100

    #maxSize = 100
    #maxTestingSize = 100

    #maxSize = 128
    #maxTestingSize = 50

    celebaTraining = celebaFiles[0:maxSize]
    celebaTesting = celebaFiles[maxSize:maxSize + maxTestingSize]

    celebaTrainSet = [GetImage_cv(
        sample_file,
        input_height=128,
        input_width=128,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in celebaTraining]
    celebaTrainSet = np.array(celebaTrainSet)

    celebaTestSet = [GetImage_cv(
        sample_file,
        input_height=128,
        input_width=128,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in celebaTesting]

    celebaTestSet = np.array(celebaTestSet)

    celebaTrainSet = np.transpose(celebaTrainSet, (0, 3, 1, 2))
    celebaTestSet = np.transpose(celebaTestSet, (0, 3, 1, 2))

    return celebaTrainSet,celebaTestSet


def Load_CelebA_Files():
    img_path = glob.glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
    # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    celebaFiles = data_files
    maxSize = 50000
    maxTestingSize = 5000

    #maxSize = 100
    #maxTestingSize = 100

    #maxSize = 128
    #maxTestingSize = 50

    celebaTraining = celebaFiles[0:maxSize]
    celebaTesting = celebaFiles[maxSize:maxSize + maxTestingSize]

    return celebaTraining,celebaTesting

    celebaTrainSet = [GetImage_cv(
        sample_file,
        input_height=128,
        input_width=128,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in celebaTraining]
    celebaTrainSet = np.array(celebaTrainSet)

    celebaTestSet = [GetImage_cv(
        sample_file,
        input_height=128,
        input_width=128,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in celebaTesting]

    celebaTestSet = np.array(celebaTestSet)

    celebaTrainSet = np.transpose(celebaTrainSet, (0, 3, 1, 2))
    celebaTestSet = np.transpose(celebaTestSet, (0, 3, 1, 2))

    return celebaTrainSet,celebaTestSet

def Load_CelebAHQ256_Files():
    img_path = glob.glob('../celeba_hq_256/*.jpg')  # 获取新文件夹下所有图片
    # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    celebaFiles = data_files
    maxSize = 25000
    maxTestingSize = 5000

    #maxSize = 100
    #maxTestingSize = 100

    #maxSize = 128
    #maxTestingSize = 50

    celebaTraining = celebaFiles
    celebaTesting = celebaFiles[maxSize:maxSize + maxTestingSize]

    return celebaTraining,celebaTesting

def Load_3DChair_Files():
    file_dir = "../rendered_chairs/"
    files = file_name2(file_dir)
    data_files = files
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    chairFiles = data_files

    maxSize = 50000
    maxTestingSize = 5000

    #maxSize = 100
    #maxTestingSize = 100

    #maxSize = 128
    #maxTestingSize = 50

    chairTraining = chairFiles[0:maxSize]
    chairTesting = chairFiles[maxSize:maxSize + maxTestingSize]

    return chairTraining,chairTesting

    image_size = 64

    chairTraining = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                     for batch_file in chairTraining]

    chairTesting = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                    for batch_file in chairTesting]

    chairTraining = np.transpose(chairTraining, (0, 3, 1, 2))
    chairTesting = np.transpose(chairTesting, (0, 3, 1, 2))

    return chairTraining,chairTesting

def Load_CelebA():
    img_path = glob.glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
    # img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
    data_files = img_path
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    celebaFiles = data_files
    maxSize = 50000
    maxTestingSize = 5000

    #maxSize = 100
    #maxTestingSize = 100

    #maxSize = 128
    #maxTestingSize = 50

    celebaTraining = celebaFiles[0:maxSize]
    celebaTesting = celebaFiles[maxSize:maxSize + maxTestingSize]

    celebaTrainSet = [GetImage_cv(
        sample_file,
        input_height=128,
        input_width=128,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in celebaTraining]
    celebaTrainSet = np.array(celebaTrainSet)

    celebaTestSet = [GetImage_cv(
        sample_file,
        input_height=128,
        input_width=128,
        resize_height=64,
        resize_width=64,
        crop=True)
        for sample_file in celebaTesting]

    celebaTestSet = np.array(celebaTestSet)

    celebaTrainSet = np.transpose(celebaTrainSet, (0, 3, 1, 2))
    celebaTestSet = np.transpose(celebaTestSet, (0, 3, 1, 2))

    return celebaTrainSet,celebaTestSet

def Load_3DChair():
    file_dir = "../rendered_chairs/"
    files = file_name2(file_dir)
    data_files = files
    data_files = sorted(data_files)
    data_files = np.array(data_files)  # for tl.iterate.minibatches
    chairFiles = data_files

    maxSize = 50000
    maxTestingSize = 5000

    #maxSize = 100
    #maxTestingSize = 100

    #maxSize = 128
    #maxTestingSize = 50

    chairTraining = chairFiles[0:maxSize]
    chairTesting = chairFiles[maxSize:maxSize + maxTestingSize]

    image_size = 64

    chairTraining = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                     for batch_file in chairTraining]

    chairTesting = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                    for batch_file in chairTesting]

    chairTraining = np.transpose(chairTraining, (0, 3, 1, 2))
    chairTesting = np.transpose(chairTesting, (0, 3, 1, 2))

    return chairTraining,chairTesting

def GiveFashion32_Tanh():

    mnistName = "Fashion"
    data_X, data_y = load_mnist_tanh(mnistName)

    # data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test


def GiveFashion32():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)

    # data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i], size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test

def Split_DataSet_CIFAR100(dataset,datasety):
    minArr = []
    maxArr = []
    for i in range(20):
        min1 = i * 5
        max1 = (i+1)*5-1

        minArr.append(min1)
        maxArr.append(max1)

    totalArr = []
    totalArr2 = []
    TArr = []
    for t1 in range(20):
        newArr = []
        totalArr.append(newArr)

        newArr2 = []
        totalArr2.append(newArr2)

    count = np.shape(dataset)[0]
    for i in range(count):
        x = dataset[i]
        y = datasety[i]

        for j in range(20):
            min1 = minArr[j]
            max1 = maxArr[j]
            if y >= min1 and y <= max1:
                totalArr[j].append(x)
                totalArr2[j].append(y)
                break

    arr1 = []
    arr2 = []
    for i in range(20):
        tarr = totalArr[i]
        tarry = totalArr2[i]
        count = np.shape(tarr)[0]
        for j in range(count):
            x = tarr[j]
            y = tarry[j]
            arr1.append(x)
            arr2.append(y)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    return arr1,arr2

def Split_DataSet_CIFAR100_Testing_New_10(dataset,datasety):
    minArr = []
    maxArr = []
    for i in range(10):
        min1 = i * 10
        max1 = (i+1)*10-1

        minArr.append(min1)
        maxArr.append(max1)

    totalArr = []
    totalArr2 = []
    TArr = []
    for t1 in range(10):
        newArr = []
        totalArr.append(newArr)

        newArr2 = []
        totalArr2.append(newArr2)

    count = np.shape(dataset)[0]
    for i in range(count):
        x = dataset[i]
        y = datasety[i]
        x = x /255.0

        for j in range(10):
            min1 = minArr[j]
            max1 = maxArr[j]
            if y >= min1 and y <= max1:
                y1 = y - min1
                totalArr[j].append(x)
                totalArr2[j].append(y1)
                break

    return totalArr,totalArr2


def Split_DataSet_CIFAR100_Testing_New(dataset,datasety):
    minArr = []
    maxArr = []
    for i in range(20):
        min1 = i * 5
        max1 = (i+1)*5-1

        minArr.append(min1)
        maxArr.append(max1)

    totalArr = []
    totalArr2 = []
    TArr = []
    for t1 in range(20):
        newArr = []
        totalArr.append(newArr)

        newArr2 = []
        totalArr2.append(newArr2)

    count = np.shape(dataset)[0]
    for i in range(count):
        x = dataset[i]
        y = datasety[i]
        x = x /255.0

        for j in range(20):
            min1 = minArr[j]
            max1 = maxArr[j]
            if y >= min1 and y <= max1:
                y1 = y - min1
                totalArr[j].append(x)
                totalArr2[j].append(y1)
                break


    return totalArr,totalArr2


def Split_DataSet_CIFAR100_Testing(dataset,datasety):
    minArr = []
    maxArr = []
    for i in range(20):
        min1 = i * 5
        max1 = (i+1)*5-1

        minArr.append(min1)
        maxArr.append(max1)

    totalArr = []
    totalArr2 = []
    TArr = []
    for t1 in range(20):
        newArr = []
        totalArr.append(newArr)

        newArr2 = []
        totalArr2.append(newArr2)

    count = np.shape(dataset)[0]
    for i in range(count):
        x = dataset[i]
        y = datasety[i]
        x = x /255.0

        for j in range(20):
            min1 = minArr[j]
            max1 = maxArr[j]
            if y >= min1 and y <= max1:
                totalArr[j].append(x)
                totalArr2[j].append(y)
                break


    return totalArr,totalArr2

def Split_dataset_by10(x,y):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7 = []
    arr8 = []
    arr9 = []
    arr10 = []

    labelArr1 = []
    labelArr2 = []
    labelArr3 = []
    labelArr4 = []
    labelArr5 = []
    labelArr6 = []
    labelArr7 = []
    labelArr8 = []
    labelArr9 = []
    labelArr10 = []

    n = np.shape(x)[0]
    for i in range(n):
        data1 = x[i]
        label1 = y[i]
        if label1[0] == 1:
            arr1.append(data1)
            labelArr1.append(label1)

        elif label1[1] == 1:
            arr2.append(data1)
            labelArr2.append(label1)

        elif label1[2] == 1:
            arr3.append(data1)
            labelArr3.append(label1)
        elif label1[3] == 1:
            arr4.append(data1)
            labelArr4.append(label1)
        elif label1[4] == 1:
            arr5.append(data1)
            labelArr5.append(label1)
        elif label1[5] == 1:
            arr6.append(data1)
            labelArr6.append(label1)
        elif label1[6] == 1:
            arr7.append(data1)
            labelArr7.append(label1)
        elif label1[7] == 1:
            arr8.append(data1)
            labelArr8.append(label1)
        elif label1[8] == 1:
            arr9.append(data1)
            labelArr9.append(label1)
        elif label1[9] == 1:
            arr10.append(data1)
            labelArr10.append(label1)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    arr4 = np.array(arr4)
    arr5 = np.array(arr5)
    arr6 = np.array(arr6)
    arr7 = np.array(arr7)
    arr8 = np.array(arr8)
    arr9 = np.array(arr9)
    arr10 = np.array(arr10)

    labelArr1 = np.array(labelArr1)
    labelArr2 = np.array(labelArr2)
    labelArr3 = np.array(labelArr3)
    labelArr4 = np.array(labelArr4)
    labelArr5 = np.array(labelArr5)
    labelArr6 = np.array(labelArr6)
    labelArr7 = np.array(labelArr7)
    labelArr8 = np.array(labelArr8)
    labelArr9 = np.array(labelArr9)
    labelArr10 = np.array(labelArr10)


    return arr1, labelArr1, arr2, labelArr2, arr3, labelArr3, arr4, labelArr4, arr5, labelArr5,arr6, labelArr6,arr7, labelArr7,arr8, labelArr8,arr9, labelArr9,arr10, labelArr10

def Split_dataset_by5(x,y):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    labelArr1 = []
    labelArr2 = []
    labelArr3 = []
    labelArr4 = []
    labelArr5 = []

    n = np.shape(x)[0]
    for i in range(n):
        data1 = x[i]
        label1 = y[i]
        if label1[0] == 1 or label1[1] == 1:
            arr1.append(data1)
            labelArr1.append(label1)

        if label1[2] == 1 or label1[3] == 1:
            arr2.append(data1)
            labelArr2.append(label1)

        if label1[4] == 1 or label1[5] == 1:
            arr3.append(data1)
            labelArr3.append(label1)

        if label1[6] == 1 or label1[7] == 1:
            arr4.append(data1)
            labelArr4.append(label1)

        if label1[8] == 1 or label1[9] == 1:
            arr5.append(data1)
            labelArr5.append(label1)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    arr4 = np.array(arr4)
    arr5 = np.array(arr5)

    labelArr1 = np.array(labelArr1)
    labelArr2 = np.array(labelArr2)
    labelArr3 = np.array(labelArr3)
    labelArr4 = np.array(labelArr4)
    labelArr5 = np.array(labelArr5)
    return arr1,labelArr1,arr2,labelArr2,arr3,labelArr3,arr4,labelArr4,arr5,labelArr5

def load_mnist_tanh(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    X = X / 127.5 -1

    return X , y_vec


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def load_mnist_256(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X,y_vec


def GetSVHN_DataSet(isResize=False):
    file1 = 'data/svhn_train.mat'
    file2 = 'data/svhn_test.mat'
    train_data = sio.loadmat(file1)
    test_data = sio.loadmat(file2)

    x_train_hv = train_data['X']
    y_train_hv = train_data['y']
    x_test_hv = test_data["X"]
    y_test_hv = test_data["y"]

    x_train_hv = x_train_hv.transpose(3, 0, 1, 2)
    x_test_hv = x_test_hv.transpose(3, 0, 1, 2)

    if isResize:
        x_train_hv = tf.image.resize_images(x_train_hv, (28, 28))
        x_test_hv = tf.image.resize_images(x_test_hv, (28, 28))
        x_train_hv = tf.image.rgb_to_grayscale(x_train_hv)
        x_test_hv = tf.image.rgb_to_grayscale(x_test_hv)

        x_train_hv = tf.Session().run(x_train_hv)
        x_test_hv = tf.Session().run(x_test_hv)

    for h1 in range(np.shape(y_test_hv)[0]):
        y_test_hv[h1] = y_test_hv[h1] - 1
    for h1 in range(np.shape(y_train_hv)[0]):
        y_train_hv[h1] = y_train_hv[h1] - 1

    x_train_hv = x_train_hv.astype('float32') / 255
    x_test_hv = x_test_hv.astype('float32') / 255

    # y_test_hv = keras.utils.to_categorical(y_test_hv)
    return x_train_hv, y_train_hv, x_test_hv, y_test_hv


def GiveMNIST_SVHN():
    mnistName = "mnist"
    data_X, data_y = load_mnist(mnistName)

    #data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i],size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    myTest = mnist_train_x[0:64]

    #ims("results/" + "gggg" + str(0) + ".jpg", merge2(myTest[:64], [8, 8]))


    x_train, y_train, x_test, y_test = GetSVHN_DataSet()
    y_train = np.eye(10, dtype=np.uint8)[y_train]#keras.utils.to_categorical(y_train)
    y_test = np.eye(10, dtype=np.uint8)[y_test]#keras.utils.to_categorical(y_test)

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test,x_train,y_train,x_test,y_test


def GiveMNIST_SVHN_Tanh():
    mnistName = "MNIST"
    data_X, data_y = load_mnist_tanh(mnistName)

    #data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i],size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    myTest = mnist_train_x[0:64]

    #ims("results/" + "gggg" + str(0) + ".jpg", merge2(myTest[:64], [8, 8]))


    x_train, y_train, x_test, y_test = GetSVHN_DataSet()
    y_train = np.eye(10, dtype=np.uint8)[y_train]#keras.utils.to_categorical(y_train)
    y_test = np.eye(10, dtype=np.uint8)[y_test]#keras.utils.to_categorical(y_test)

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test,x_train,y_train,x_test,y_test


def GiveMNIST_SVHN_256():
    mnistName = "mnist"
    data_X, data_y = load_mnist_256(mnistName)

    #data_X = np.expand_dims(data_X, axis=3)
    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i],size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    myTest = mnist_train_x[0:64]

    #ims("results/" + "gggg" + str(0) + ".jpg", merge2(myTest[:64], [8, 8]))


    x_train, y_train, x_test, y_test = GetSVHN_DataSet()
    y_train = np.eye(10, dtype=np.uint8)[y_train]#keras.utils.to_categorical(y_train)
    y_test = np.eye(10, dtype=np.uint8)[y_test]#keras.utils.to_categorical(y_test)

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test,x_train,y_train,x_test,y_test


def Split_dataset(x,y,n_label):
    y = np.argmax(y,axis=1)
    n_each = n_label / 10
    isRun = True
    x_train = []
    y_train = []
    index = np.zeros(10)
    while(isRun):
        a = random.randint(0, np.shape(x)[0])-1
        x1 = x[a]
        y1 = y[a]
        if index[y1] < n_each:
            x_train.append(x1)
            y_train.append(y1)
            index[y1] = index[y1]+1
        isOk1 = True
        for i in range(10):
            if index[i] < n_each:
                isOk1 = False
        if isOk1:
            break

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train,y_train


def Give_InverseFashion32():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X,(-1,28,28))

    data_X = np.concatenate((data_X, data_X, data_X), axis=-1)

    size = (int(32), int(32))
    myArr = []
    for i in range(np.shape(data_X)[0]):
        image = cv2.resize(data_X[i],size, interpolation=cv2.INTER_AREA)
        myArr.append(image)

    data_X = np.array(myArr)

    for i in range(np.shape(data_X)[0]):
        for k1 in range(32):
            for k2 in range(32):
                for k3 in range(3):
                    data_X[i,k1,k2,k3] = 1.0 - data_X[i,k1,k2,k3]

    data_X = np.reshape(data_X,(-1,32,32,3))

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    return x_train,y_train,x_test,y_test

def Give_InverseFashion():
    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)
    data_X = np.reshape(data_X,(-1,28,28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i,k1,k2] = 1.0 - data_X[i,k1,k2]

    data_X = np.reshape(data_X,(-1,28,28,1))
    return data_X,data_y

def Give_InverseMNIST32():
    mnistName = "mnist"
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

    data_X = np.array(myArr)

    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]
    y_train = data_y[0:60000]
    y_test = data_y[60000:70000]

    mnist_train_x = x_train
    mnist_train_label = y_train
    mnist_test = x_test
    mnist_label_test = y_test

    return mnist_train_x,mnist_train_label,mnist_test,mnist_label_test
