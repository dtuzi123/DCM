import numpy as np

from NetworkModels.Teacher_Model_ import Teacher,Balance_Teacher
from NetworkModels.VAE_Model_ import StudentModel,Balance_StudentModel
import torch.nn as nn
from datasets.Data_Loading import *
from datasets.Fid_evaluation import *
import torch as th

class DynamicTeacherStudent(nn.Module):

    def __init__(self,name,device,input_size):
        super(DynamicTeacherStudent, self).__init__()

        self.expansion_threshold = 100
        self.selection_score = 0
        self.selectedTeacher = 0

        self.input_size = input_size
        self.teacher = Balance_Teacher(input_size)
        self.student = Balance_StudentModel(device,input_size)
        self.device = device

        self.teacherArr = []
        self.teacherArr.append(self.teacher)
        self.selectedTeacher = self.teacher

    def CheckExpansion(self,newTask):
        maxCount = 500
        dataset = newTask[0:maxCount]

        arr = []
        for i in range(np.shape(self.teacherArr)[0]):
            g1 = self.teacherArr.Give_Generation(maxCount)
            fid = calculate_fid_given_paths_Byimages(g1, dataset, 50, self.device, 2048)
            arr.append(fid)

        minScore = np.min(arr)
        minIndex = np.argmin(arr)

        if minScore > self.expansion_threshold:#Check the expansion
            #Perform the expansion
            newTeacher = Balance_Teacher(self.input_size)
            self.teacherArr.append(newTeacher)
            self.selectedTeacher = newTeacher
            self.selection_score = 0
        else:
            #Perform the component selection
            self.selectedTeacher = self.teacherArr[minIndex]
            self.selection_score = 1

    def Train(self,Tepoch,Sepoch,data,generatedData):

        #Check the expansion
        dataX = data

        if self.selection_score == 0:
            self.selectedTeacher.Train_Self(Tepoch,data)
            #train the student
            count = np.shape(self.dynamicTeacher.teacherArr)[0]
            batchSize = int(self.batch_size / count)
            totalCount = np.shape(dataX)[0] / batchSize

            for i in range(totalCount):
                arr2 = dataX[i * batchSize:(i + 1) * batchSize]
                for j in range(np.shape(self.dynamicTeacher.teacherArr)[0]):
                    newa = self.dynamicTeacher.teacherArr[j].Give_Generation(batchSize)
                    arr2 = th.cat([arr2, newa], dim=0)
                self.selectedTeacher.student.Train_One(arr2)
        else:
            self.selectedTeacher.Train_Self(Tepoch, data)
            # train the student
            count = np.shape(self.dynamicTeacher.teacherArr)[0]
            batchSize = int(self.batch_size / count)
            totalCount = np.shape(dataX)[0] / batchSize

            for i in range(totalCount):
                arr2 = dataX[i * batchSize:(i + 1) * batchSize]
                for j in range(np.shape(self.dynamicTeacher.teacherArr)[0]):
                    newa = self.dynamicTeacher.teacherArr[j].Give_Generation(batchSize)
                    arr2 = th.cat([arr2, newa], dim=0)
                self.selectedTeacher.student.Train_One(arr2)

        if np.shape(generatedData)[0] == 0:
            self.teacher.Train_Self(Tepoch,data)
            self.student.Train_Self(Sepoch,data)
        else:
            self.teacher.Train_Self_(Tepoch,data,generatedData)
            self.student.Train_Self_(Sepoch,data,generatedData)


