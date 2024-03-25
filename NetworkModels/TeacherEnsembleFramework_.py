import numpy as np
from improved_diffusion import dist_util, logger

#from NetworkModels.Teacher_Model_ import Teacher,Balance_Teacher
from NetworkModels.Teacher_Model_NoMPI_ import Balance_Teacher_NoMPI
from NetworkModels.VAE_Model_ import Balance_StudentModel
import torch.nn as nn
from NetworkModels.TFCL_Teacher_ import *
from improved_diffusion.train_util_balance_NoMPI_MultiGPU import *
from NetworkModels.VAE_Model_ import *
import random
import torch.distributions as td
from models.VAE256 import *
from NetworkModels.DynamicDiffusionMixture_ import *
from NetworkModels.MMD_Lib import *
from models.VAE256 import *
from NetworkModels.MyClassifier import *
from NetworkModels.MemoryUnitFramework_ import *
import numpy as np
from datasets.Fid_evaluation import *

class TeacherEnsembleFramework(MemoryUnitFramework):
    def __init__(self,name,device,input_size):
        super(MemoryUnitFramework, self).__init__(name,device,input_size)

        self.input_size = input_size
        self.device = device
        self.trainingCount = 0
        self.trainingUpdate = 4
        self.GeneratingBatchSampleSize = 64
        self.batchTrainStudent_size = 64
        self.isTrainer = 0
        self.teacherArray = []
        self.autoencoderArr = []

        self.isExpansion = True

        self.student = StudentModel(device,input_size)#Autoencoder(device, input_size)
        self.autoencoderArr.append(self.student)

        self.currentComponent = 0
        self.batch_size = 64

        self.resultMatrix = []
        self.unitCount = 0

        self.memoryUnits = []
        self.currentMemory = []
        self.threshold = 0.02

        self.maxMemorySize = 2000
        self.diversityThreshold = 0.02
        self.expansionThreshold = 140
        self.memoryUnitSize = 64
        self.currentTrainingIndex = 0
        self.maxTrainingTime = 200
        self.currentTrainingTime = 0

    def Transfer_To_Numpy(self,sample):
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)
        mySamples = mySamples[0]
        return mySamples

    def CheckModelExpansion_TaskKnown(self,newDataset):
        isExpansion = True
        index = 0

        genCount = 1000
        newDataset2 = newDataset[0:genCount]
        newDataset2 = torch.tensor(newDataset2).cuda().to(device=self.device, dtype=torch.float)

        arr = []
        for i in range(np.shape(self.teacherArray)[0]):
            gan1 = self.teacherArray[i].GenerateImages(genCount)
            #gan1 = self.Transfer_To_Numpy(gan1)

            fid1 = calculate_fid_given_paths_Byimages(newDataset2, gan1, 50, self.device, 2048)
            arr.append(fid1)

        minscore = np.min(arr)
        if minscore > self.expansionThreshold:
            #Perform the expansion
            newComponent = self.Create_NewComponent()
            self.currentComponent = newComponent
        else:
            #Perform the expert selection
            index = np.argmin(arr)
            self.currentComponent = self.teacherArray[index]
            isExpansion = False

        return minscore,index,arr,isExpansion

    def TrainStudent_Numpy(self,epoch,memory):
        self.student.Train_Self_Single_Beta3_Numpy(epoch,memory)

    def TrainStudent_Balance_Numpy(self,epoch,generatedData,memory):
        self.student.Train_Self_Single_Beta3_Balance_Numpy(epoch,generatedData,memory)


    def TrainStudent(self, epoch, memory):
        # using the KD
        self.student.training_step2_WithBeta()
        self.student.Train_Self_Single_Beta3(epoch, memory)
