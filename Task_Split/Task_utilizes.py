import numpy as np


def Split_DataSet_CIFAR100_By20(dataset,datasety):
    minArr = []
    maxArr = []
    for i in range(5):
        min1 = i * 20
        max1 = (i+1)*20-1

        minArr.append(min1)
        maxArr.append(max1)

    totalArr = []
    totalArr2 = []
    TArr = []
    for t1 in range(5):
        newArr = []
        totalArr.append(newArr)

        newArr2 = []
        totalArr2.append(newArr2)

    count = np.shape(dataset)[0]
    for i in range(count):
        x = dataset[i]
        y = datasety[i]

        for j in range(5):
            min1 = minArr[j]
            max1 = maxArr[j]
            if y >= min1 and y <= max1:
                totalArr[j].append(x)
                totalArr2[j].append(y)
                break

    return totalArr,totalArr2
    '''
    arr1 = []
    arr2 = []
    for i in range(5):
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
    '''


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

def Split_Dataset_Five(train_data,train_labels,test_data,test_labels):
    arr1, labelArr1, arr2, labelArr2, arr3, labelArr3, arr4, labelArr4, arr5, labelArr5 = Split_dataset_by5(train_data,
                                                                                                            train_labels)
    arr1_test, labelArr1_test, arr2_test, labelArr2_test, arr3_test, labelArr3_test, arr4_test, labelArr4_test, arr5_test, labelArr5_test = Split_dataset_by5(
        test_data,
        test_labels)

    totalSetX = []
    totalSetX.append(arr1)
    totalSetX.append(arr2)
    totalSetX.append(arr3)
    totalSetX.append(arr4)
    totalSetX.append(arr5)

    totalSetY = []
    totalSetY.append(labelArr1)
    totalSetY.append(labelArr2)
    totalSetY.append(labelArr3)
    totalSetY.append(labelArr4)
    totalSetY.append(labelArr5)

    totalTestX = []
    totalTestX.append(arr1_test)
    totalTestX.append(arr2_test)
    totalTestX.append(arr3_test)
    totalTestX.append(arr4_test)
    totalTestX.append(arr5_test)

    totalTestY = []
    totalTestY.append(labelArr1_test)
    totalTestY.append(labelArr2_test)
    totalTestY.append(labelArr3_test)
    totalTestY.append(labelArr4_test)
    totalTestY.append(labelArr5_test)

    return totalSetX,totalSetY,totalTestX,totalTestY