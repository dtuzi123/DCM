
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


class DynamicMixture256(DynamicDiffusionMixture):
    def __init__(self,name,device,input_size,modelType,originalInputSize):
        super(DynamicDiffusionMixture, self).__init__()

        self.input_size = input_size
        self.device = device
        self.trainingCount = 0
        self.trainingUpdate = 4
        self.GeneratingBatchSampleSize = 64
        self.batchTrainStudent_size = 64
        self.isTrainer = 0
        self.teacherArray = []
        self.autoencoderArr = []
        self.memorybuffer = []
        self.OriginalInputSize = originalInputSize
        self.currentTrainingTime = 0

        if modelType == "GAN":
            print("GAN")
        else:
            teacher = Autoencoder(device, input_size)
            teacher.OriginalInputSize = self.OriginalInputSize
            self.currentComponent = teacher
            self.teacherArray.append(teacher)
            self.student = Autoencoder(device, input_size)
            self.student.OriginalInputSize = self.OriginalInputSize
            #print(self.currentComponent.input_size)

    def Create_NewTeacher(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        teacher = Autoencoder(device, self.input_size)
        self.currentComponent = teacher
        self.teacherArray.append(teacher)

    def Check_Expansion_Cpu(self):

        arr = []
        for i in range(np.shape(self.teacherArr)[0]-1):
            print("build")
            memory = self.memorybuffer[0:1000]
            memory = memory
            memory = torch.FloatTensor(memory)

            generated = self.teacherArr[i].GiveGeneration_Cpu(np.shape(memory)[0])
            #generated = generated.cpu()

            fid = calculate_fid_given_paths_Byimages(memory, generated, 50, self.device, 2048)
            #fid = fid.cpu().detach().numpy()
            arr.append(fid)

        minvalue = np.min(arr)
        print(minvalue)

        if minvalue > self.threshold:
            print("Build")
            self.Create_NewTeacher()


    def TrainTeacher(self,epoch,memoryBuffer):
        self.currentComponent.Train_Cpu_WithFiles(epoch, memoryBuffer)

    def TrainStudent_Cpu(self,epoch,memory):
        # using the KD
        self.student.Train_Self_Single_Beta3_Cpu(epoch,memory)


    def GiveGenerationByTeacher(self, num):
        count = np.shape(self.teacherArray)[0]
        t = int(num / 2)
        arr = []
        for i in range(t):
            index = random.randint(1, count) - 1
            new1 = self.teacherArray[index].vae.sample_with_noise(2,self.device)
            if np.shape(arr)[0] == 0:
                arr = new1
            else:
                arr = torch.cat([arr, new1], 0)
        return arr


    def Give_GenerationFromTeacher_Cpu(self,num):

        with torch.no_grad():
            count = np.shape(self.teacherArray)[0]
            t = int(num / 2)
            arr = []
            for i in range(t):
                index = random.randint(1,count) - 1
                new1 = self.teacherArray[index].Give_GenerationsWithN(2)

                new1 = new1.unsqueeze(0).cuda().cpu()
                new1 = np.array(new1)
                new1 = new1[0]

                if np.shape(arr)[0] == 0:
                    arr = new1
                else:
                    #arr = torch.cat([arr,new1],0)
                    arr = np.concatenate((arr,new1),0)
            return arr
