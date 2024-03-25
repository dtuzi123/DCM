import numpy as np

#from NetworkModels.Teacher_Model_ import Teacher,Balance_Teacher
from NetworkModels.Teacher_Model_NoMPI_ import Balance_Teacher_NoMPI
from NetworkModels.VAE_Model_ import Balance_StudentModel
import torch.nn as nn
from NetworkModels.TFCL_Teacher_ import *
from improved_diffusion.train_util_balance_NoMPI_MultiGPU import *
from NetworkModels.VAE_Model_ import *

class TFCL_DynamicVAEModel(nn.Module):
    def __init__(self,name,device,input_size):
        super(TFCL_DynamicVAEModel, self).__init__()

        self.input_size = input_size
        self.currentComponent = TFCL_StudentModel(device,input_size)
        self.componentList = []
        self.componentList.append(self.currentComponent)

        self.device = device
        self.trainingCount = 0
        self.trainingUpdate = 4
        self.GeneratingBatchSampleSize = 64
        self.batch_size = 64
        self.isTrainer = 0

    def AddNewComponent(self):
        a = TFCL_StudentModel(self.device,self.input_size)
        self.currentComponent = a
        self.componentList.append(a)

    def SelectComponent_BySample(self,singleData):
        arr = []
        for i in range(np.shape(self.componentList)[0]):
            loss1 = self.componentList[i].ComputerLoss(singleData)
            loss1 = loss1['loss']
            loss1 = loss1.cpu().detach().numpy()
            arr.append(loss1)
        minindex = np.argmin(arr)
        return minindex

    def Train_SelectedComponent(self,epoch,data,component):
        component.Train_Self(epoch,data)

    def GiveReconstructionBatch(self,batch):
        arr = []
        for i in range(np.shape(batch)[0]):
            sample = batch[i]
            sample = torch.reshape(sample, (1, 3, self.input_size, self.input_size))
            sample = torch.cat([sample, sample], 0)

            index = self.SelectComponent_BySample(sample)

            reco = self.componentList[index].Give_ReconstructionSingle(sample)
            reco = reco[0]
            reco = torch.reshape(reco, (1, 3, self.input_size, self.input_size))
            if np.shape(arr)[0] == 0:
                arr = reco
            else:
                arr = torch.cat([arr,reco],0)
        return arr

    def GiveReconstructionFromOriginalImages(self,data):
        arr = []
        count = int(np.shape(data)[0]/self.batch_size)
        for i in range(count):
            batch = data[i*self.batch_size:(i+1)*self.batch_size]
            batch = th.tensor(batch).cuda().to(device=self.device, dtype=th.float)
            reco = self.GiveReconstructionBatch(batch)
            if np.shape(arr)[0] == 0:
                arr = reco
            else:
                arr = th.cat([arr, reco], 0)
        return arr

    def GiveReconstruction(self,data):
        arr = []

        count = int(np.shape(data)[0]/self.batch_size)
        for i in range(count):
            batch = data[i*self.batch_size:(i+1)*self.batch_size]
            reco = self.GiveReconstructionBatch(batch)
            if np.shape(arr)[0] == 0:
                arr = reco
            else:
                arr = torch.cat([arr,reco],0)
        return arr


    def GiveMixGeneration(self,num):
        count = int(num / self.batch_size)
        t2 = int(self.batch_size/np.shape(self.componentList)[0])

        arr = []
        for i in range(count):
            for j in range(np.shape(self.componentList)[0]):
                x1 = self.componentList[j].Generation(t2)
                if np.shape(arr)[0] == 0:
                    arr = x1
                else:
                    arr = torch.cat([arr,x1],0)

        return arr

