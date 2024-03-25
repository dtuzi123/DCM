
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

class MemoryUnitFramework(DynamicDiffusionMixture):
    def __init__(self,name,device,input_size):
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

        '''
        if input_size == 256:
            self.student = Autoencoder256()
        else:
            self.student = Autoencoder(device,input_size)
        '''

        self.student = Autoencoder(device, input_size)
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
        self.expansionThreshold = 0.02
        self.memoryUnitSize = 64
        self.currentTrainingIndex = 0
        self.maxTrainingTime = 200
        self.currentTrainingTime = 0

    def TrainStudent_Numpy(self,epoch,memory):
        self.student.Train_Self_Single_Beta3_Numpy(epoch,memory)

    def GiveMemorySize(self):
        count = (self.unitCount+1)*self.memoryUnitSize
        return count

    def SampleSelection(self,dataBatch):
        if np.shape(self.currentMemory)[0] == 0:
            return dataBatch

        memory = self.currentMemory
        memory2 = dataBatch

        #print(np.shape(memory))
        #print(np.shape(memory2))

        if np.shape(self.autoencoderArr)[0] > 1:

            lossArr = []
            lossArr2 = []

            for i in range(np.shape(self.autoencoderArr)[0]):
                a1 = self.autoencoderArr[i].vae.GiveLoss(self.currentMemory)
                a2 = self.autoencoderArr[i].vae.GiveLoss(memory2)

                a1 = np.array(a1)
                a2 = np.array(a2)

                if np.shape(lossArr)[0] == 0:
                    lossArr = a1
                    lossArr2 = a2
                else:
                    lossArr = a1 + lossArr
                    lossArr2 = a2 + lossArr2
        else:
            lossArr = self.student.vae.GiveLoss(self.currentMemory)
            lossArr2 = self.student.vae.GiveLoss(memory2)

        #print(np.shape(lossArr))
        #print(lossArr)

        memory = th.cat([memory,memory2],0)
        lossArr = np.concatenate((lossArr,lossArr2),0)
        lossArr = lossArr*-1

        index = np.argsort(lossArr)

        memory = memory[index]
        memory = memory[0:self.memoryUnitSize]
        return memory

    def LoadCACDFromPath(self,path):
        batch = [GetImage_cv(
            sample_file,
            input_height=self.originalInputSize,
            input_width=self.originalInputSize,
            resize_height=self.input_size,
            resize_width=self.input_size,
            crop=False)
            for sample_file in path]
        return batch

    def CheckExpansionByMMD(self,batch):
        isState = False

        arr = []
        for j in range(np.shape(self.teacherArray)[0]):
            generated = self.teacherArray[j].GenerateImages(64)
            mmd = self.Calculate_MMD(batch,generated)
            arr.append(mmd)

        score = np.min(arr)
        #print("Expansion")
        print(score)
        if score > self.expansionThreshold:
            print("Build")
            newComponent = self.Create_NewComponent()
            self.currentComponent = newComponent
            a1 = Autoencoder(self.device, self.input_size)
            self.autoencoderArr.append(a1)

            self.memoryUnits.clear()
            self.memoryUnits = []
            self.unitCount = 0
            isState = True

        return isState

    def CheckExpansionByMMDAndState(self,batch):
        arr = []
        for j in range(np.shape(self.teacherArray)[0]):
            generated = self.teacherArray[j].GenerateImages(64)
            mmd = self.Calculate_MMD(batch,generated)
            arr.append(mmd)

        score = np.min(arr)
        print("Expansion")
        print(score)
        if score > self.expansionThreshold:
            print("Build")
            newComponent = self.Create_NewComponent()
            self.currentComponent = newComponent
            a1 = Autoencoder(self.device, self.input_size)
            self.autoencoderArr.append(a1)

            self.memoryUnits.clear()
            self.memoryUnits = []
            self.unitCount = 0

    def CheckExpansionByMMDAndState_Files(self,batch):
        arr = []
        for j in range(np.shape(self.teacherArray)[0]):
            oldbatchimages = self.LoadCACDFromPath(batch)
            oldbatchimages = np.array(oldbatchimages)
            oldbatchimages = oldbatchimages.transpose(0, 3, 1, 2)
            oldbatchimages = th.tensor(oldbatchimages).cuda().to(device=self.device, dtype=torch.float)

            generated = self.teacherArray[j].GenerateImages(np.shape(batch)[0])
            mmd = self.Calculate_MMD(oldbatchimages,generated)
            arr.append(mmd)

        score = np.min(arr)
        print("Expansion")
        print(score)
        if score > self.expansionThreshold:
            print("Build")
            newComponent = self.Create_NewComponent_Cpu()
            self.currentComponent = newComponent
            a1 = Autoencoder(self.device, self.input_size)
            self.autoencoderArr.append(a1)

            self.memoryUnits.clear()
            self.memoryUnits = []
            self.unitCount = 0


    def CheckExpansionByDiversity(self):
        score = self.Calculate_Diversity()
        print("Diversity")

        if score > self.diversityThreshold:
            print("Build a new component")
            newComponent = self.Create_NewComponent()
            self.currentComponent = newComponent
            self.memoryUnits.clear()
            self.memoryUnits = []

    def Calculate_Diversity(self):

        myCount = 0
        sum = 0
        maxCount = 64
        currentGenerated = self.currentComponent.GenerateImages(maxCount)
        for i in range(maxCount):
            a = currentGenerated[i]
            for j in range(maxCount - i):
                b = currentGenerated[j]
                dis = self.Calculate_MMD_Single(a,b)
                sum = sum + dis
                myCount = myCount+1

        #sum = sum / myCount
        return sum

    def SampleSelection_Files(self,dataBatch):
        if np.shape(self.currentMemory)[0] == 0:
            return dataBatch

        memory = self.currentMemory
        memory2 = dataBatch

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #print(np.shape(memory))
        #print(np.shape(memory2))

        batchImage = self.LoadCACDFromPath(self.currentMemory)
        batchImage = th.tensor(batchImage).cuda().to(device=device, dtype=th.float)
        batchImage = batchImage.permute(0, 3, 1, 2)
        batchImage = batchImage.contiguous()

        batchImage2 = self.LoadCACDFromPath(memory2)
        batchImage2 = th.tensor(batchImage2).cuda().to(device=device, dtype=th.float)
        batchImage2 = batchImage2.permute(0, 3, 1, 2)
        batchImage2 = batchImage2.contiguous()

        lossArr = self.student.vae.GiveLoss(batchImage)
        lossArr2 = self.student.vae.GiveLoss(batchImage2)

        #print(np.shape(lossArr))
        #print(lossArr)

        memory = np.concatenate((memory,memory2),0) #th.cat([memory,memory2],0)
        lossArr = np.concatenate((lossArr,lossArr2),0)
        lossArr = lossArr*-1

        index = np.argsort(lossArr)

        memory = memory[index]
        memory = memory[0:64]

        return memory

    def GiveMemorizedSamples(self):
        if self.unitCount == 0:
            return self.currentMemory
        else:
            memory = self.currentMemory
            for i in range(self.unitCount):
                old = self.memoryUnits[i]
                memory = th.cat([memory,old],0)
            return memory

    def GiveMemorizedSamples_Files(self):
        if self.unitCount == 0:
            return self.currentMemory
        else:
            memory = self.currentMemory
            for i in range(self.unitCount):
                old = self.memoryUnits[i]
                memory = np.concatenate((memory,old),0)  #th.cat([memory,old],0)
            return memory

    def Calculate_DiscrepancyScore(self,index):
        score = 0
        current = self.memoryUnits[index]
        for i in range(self.unitCount):
            if i != index:
                other = self.memoryUnits[i]
                dis = self.Calculate_MMD(current,other)
                score += dis

        return score

    def Calculate_DiscrepancyScore_Files(self,index):
        score = 0
        current = self.memoryUnits[index]

        oldbatchimages = self.LoadCACDFromPath(current)
        oldbatchimages = np.array(oldbatchimages)
        oldbatchimages = oldbatchimages.transpose(0, 3, 1, 2)
        oldbatchimages = th.tensor(oldbatchimages).cuda().to(device=self.device, dtype=th.float)

        for i in range(self.unitCount):
            if i != index:
                other = self.memoryUnits[i]

                currentbatchimages = self.LoadCACDFromPath(other)
                currentbatchimages = np.array(currentbatchimages)
                currentbatchimages = currentbatchimages.transpose(0, 3, 1, 2)
                currentbatchimages = th.tensor(currentbatchimages).cuda().to(device=self.device, dtype=th.float)

                dis = self.Calculate_MMD(oldbatchimages,currentbatchimages)
                score += dis

        return score

    def RemoveData_FromMemory_Files(self):
        arr = []
        nodeArr1 = []
        nodeArr2 = []
        for i in range(self.unitCount):
            for j in range(i + 1, self.unitCount):
                nodeArr1.append(i)
                nodeArr2.append(j)
                old = self.memoryUnits[i]
                other = self.memoryUnits[j]

                th=torch

                oldbatchimages = self.LoadCACDFromPath(old)
                oldbatchimages = np.array(oldbatchimages)
                oldbatchimages = oldbatchimages.transpose(0, 3, 1, 2)
                oldbatchimages = th.tensor(oldbatchimages).cuda().to(device=self.device, dtype=th.float)

                currentbatchimages = self.LoadCACDFromPath(other)
                currentbatchimages = np.array(currentbatchimages)
                currentbatchimages = currentbatchimages.transpose(0, 3, 1, 2)
                currentbatchimages = th.tensor(currentbatchimages).cuda().to(device=self.device, dtype=th.float)

                mmd = self.Calculate_MMD(oldbatchimages, currentbatchimages)
                arr.append(mmd)

        index = np.argmin(arr)
        removedIndex1 = nodeArr1[index]
        removedIndex2 = nodeArr2[index]
        score1 = self.Calculate_DiscrepancyScore_Files(removedIndex1)
        score2 = self.Calculate_DiscrepancyScore_Files(removedIndex2)

        removedIndex = removedIndex1
        if score1 > score2:
            removedIndex = removedIndex2

        removedNode = self.memoryUnits[removedIndex]

        newMemoryArr = []

        for h in range(self.unitCount):
            if h != removedIndex:
                newMemoryArr.append(self.memoryUnits[h])

        # self.memoryUnits.remove(removedNode)
        self.memoryUnits = newMemoryArr
        self.unitCount = self.unitCount - 1

    def RemoveData_FromMemory(self):
        arr = []
        nodeArr1 = []
        nodeArr2 = []
        for i in range(self.unitCount):
            for j in range(i+1,self.unitCount):
                nodeArr1.append(i)
                nodeArr2.append(j)
                old = self.memoryUnits[i]
                other = self.memoryUnits[j]
                mmd = self.Calculate_MMD(old, other)
                arr.append(mmd)

        index = np.argmin(arr)
        removedIndex1 = nodeArr1[index]
        removedIndex2 = nodeArr2[index]
        score1 = self.Calculate_DiscrepancyScore(removedIndex1)
        score2 = self.Calculate_DiscrepancyScore(removedIndex2)

        removedIndex = removedIndex1
        if score1 > score2:
            removedIndex = removedIndex2

        removedNode = self.memoryUnits[removedIndex]

        newMemoryArr = []

        for h in range(self.unitCount):
            if h != removedIndex:
                newMemoryArr.append(self.memoryUnits[h])

        #self.memoryUnits.remove(removedNode)
        self.memoryUnits = newMemoryArr
        self.unitCount = self.unitCount-1

    def Check_Memory_Expansion_Files(self):
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        state = False
        measure = 0
        if self.unitCount == 0:
            self.memoryUnits.append(self.currentMemory)
            self.currentMemory = []
            self.unitCount += 1
            print("NULL")
        else:
            arr = []
            #Check the memory expansion
            for i in range(self.unitCount):
                old = self.memoryUnits[i]

                oldbatchimages = self.LoadCACDFromPath(old)
                oldbatchimages = np.array(oldbatchimages)
                oldbatchimages = oldbatchimages.transpose(0, 3, 1, 2)
                oldbatchimages = th.tensor(oldbatchimages).cuda().to(device=device, dtype=th.float)

                currentbatchimages = self.LoadCACDFromPath(self.currentMemory)
                currentbatchimages = np.array(currentbatchimages)
                currentbatchimages = currentbatchimages.transpose(0, 3, 1, 2)
                currentbatchimages = th.tensor(currentbatchimages).cuda().to(device=device, dtype=th.float)

                mmd = self.Calculate_MMD(oldbatchimages,currentbatchimages)
                arr.append(mmd)

            minV = np.min(arr)
            measure = minV
            if minV > self.threshold:
                self.memoryUnits.append(self.currentMemory)
                self.unitCount += 1
                self.currentMemory = []
                state = True

        #Check the maximum size
        if self.GiveMemorySize() > self.maxMemorySize:
            #if self.input_size != 256:
            self.RemoveData_FromMemory_Files()

        return measure,state

    def Check_Memory_Expansion(self):
        state = False
        measure = 0
        if self.unitCount == 0:
            self.memoryUnits.append(self.currentMemory)
            self.currentMemory = []
            self.unitCount += 1
        else:
            arr = []
            #Check the memory expansion
            for i in range(self.unitCount):
                old = self.memoryUnits[i]
                mmd = self.Calculate_MMD(old,self.currentMemory)
                arr.append(mmd)

            minV = np.min(arr)
            measure = minV
            if minV > self.threshold:
                self.memoryUnits.append(self.currentMemory)
                self.unitCount += 1
                self.currentMemory = []
                state = True

        #Check the maximum size
        if self.GiveMemorySize() > self.maxMemorySize:
            self.RemoveData_FromMemory()

        return measure,state

    def Check_Memory_ExpansionWithoutMRP(self):

        measure = 0
        state = False

        if self.GiveMemorySize() < self.maxMemorySize:

            measure = 0
            if self.unitCount == 0:
                self.memoryUnits.append(self.currentMemory)
                self.currentMemory = []
                self.unitCount += 1
            else:
                arr = []
                #Check the memory expansion
                for i in range(self.unitCount):
                    old = self.memoryUnits[i]
                    mmd = self.Calculate_MMD(old,self.currentMemory)
                    arr.append(mmd)

                minV = np.min(arr)
                measure = minV
                if minV > self.threshold:
                    self.memoryUnits.append(self.currentMemory)
                    self.unitCount += 1
                    self.currentMemory = []
                    state = True

        return measure,state

    def KnowledgeDistillation(self,epoch):
        student = self.teacherArray[np.shape(self.teacherArray)[0]-1]
        smallBatch = int(self.batch_size / np.shape(self.teacherArray)[0])

        for i in range(epoch):
            arr = []
            for j in range(np.shape(self.teacherArray)[0]):
                data = self.teacherArray[j].GenerateImages(smallBatch)
                if np.shape(data)[0] == 0:
                    arr = data
                else:
                    arr = th.cat([arr,data],0)

                student.Train(1, arr)

        return student

    def Calculate_MMD_Single(self,a,b):
        a = a.reshape(1,3,self.input_size,self.input_size)
        b = b.reshape(1,3,self.input_size,self.input_size)
        a = a.repeat(64,1,1,1)
        b = b.repeat(64,1,1,1)
        dis = self.Calculate_MMD(a,b)
        return dis

    def Calculate_MMD(self,dataA,dataB):
        if np.shape(self.autoencoderArr)[0] > 1:
            code1 = []
            code2 = []
            sum1 = 0
            for i in range(np.shape(self.autoencoderArr)[0]):
                a1 = self.autoencoderArr[i].vae.GiveCode(dataA)
                a2 = self.autoencoderArr[i].vae.GiveCode(dataB)
                a1 = a1.cpu().detach().numpy()
                a2 = a2.cpu().detach().numpy()
                t1 = mmd_linear(a1, a2)
                sum1 = sum1 + t1

            sum1 = sum1 / int(np.shape(self.autoencoderArr)[0])
            return sum1
        else:
            code1 = self.student.vae.GiveCode(dataA)
            code2 = self.student.vae.GiveCode(dataB)

            code1 = code1.cpu().detach().numpy()
            code2 = code2.cpu().detach().numpy()

            return mmd_linear(code1, code2)
    
    '''
    def Calculate_MMD(self,dataA,dataB):
        if np.shape(self.autoencoderArr)[0] > 1:
            code1 = []
            code2 = []
            for i in range(np.shape(self.autoencoderArr)[0]):
                a1 = self.autoencoderArr[i].vae.GiveCode(dataA)
                a2 = self.autoencoderArr[i].vae.GiveCode(dataB)
                if np.shape(code1)[0] == 0:
                    code1 = a1
                    code2 = a2
                else:
                    code1 = code1 + a1
                    code2 = code2 + a2

            code1 = code1.cpu().detach().numpy()
            code2 = code2.cpu().detach().numpy()

            return mmd_linear(code1, code2)
        else:

            code1 = self.student.vae.GiveCode(dataA)
            code2 = self.student.vae.GiveCode(dataB)

            code1 = code1.cpu().detach().numpy()
            code2 = code2.cpu().detach().numpy()

            return mmd_linear(code1, code2)
    '''

    def create_argparser(self):
        defaults = dict(
            data_dir="/scratch/fy689/improved-diffusion-main/cifar_train",
            schedule_sampler="uniform",
            lr=1e-4,
            weight_decay=0.0,
            lr_anneal_steps=0,
            batch_size=128,
            microbatch=-1,  # -1 disables microbatches
            ema_rate="0.9999",  # comma-separated list of EMA values
            log_interval=10,
            save_interval=10000,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

    def TrainStudent(self,epoch,memory):
        # using the KD
        self.student.Train(epoch,memory)

    def TrainStudent_Files(self,epoch,memory):

        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        count = int(np.shape(memory)[0] / self.batch_size)
        for i in range(epoch):
            for j in range(count):
                files = memory[j*self.batch_size:(j+1)*self.batch_size]
                batchimages = self.LoadCACDFromPath(files)
                batchimages = np.array(batchimages)
                batchimages = batchimages.transpose(0, 3, 1, 2)
                batchimages = th.tensor(batchimages).cuda().to(device=device, dtype=th.float)

                loss = self.student.vae.Update_Reco(batchimages)

        # using the KD
        #self.student.Train_Files(epoch,memory)

    def TrainingStudentFromTeacher(self,epoch,memory,isMemory):
        iterations = np.shape(memory)[0] / self.batch_size

        optimizer = self.student.optimizer
        if isMemory == True:
            for i in range(epoch):
                for j in range(iterations):
                    optimizer.zero_grad()
                    realbatch = memory[j*self.batch_size:(j+1)*self.batch_size]
                    loss = self.student.training_step(realbatch)

                    weight = 1.0 / np.shape(self.teacherArray)[0]
                    loss = loss * weight

                    for c in range(np.shape(self.teacherArray)[0]):
                        minmemory = self.teacherArray[c].memoryBuffer
                        loss2 = self.student.training_step(minmemory)
                        loss += loss2 * weight
                    loss.backward()
                    self.optimizer.step()


    def TrainStudent_Cpu(self,epoch,memory):
        # using the KD
        self.student.Train_Self_Single_Beta3_Cpu(epoch,memory)

    def TrainStudent_Cpu_WithBeta(self,epoch,memory,beta):
        # using the KD
        self.student.Train_Self_Single_Beta3_Cpu_WithBeta(epoch,memory,beta)

    def RemoveExpertFromindex(self,index):
        current = self.teacherArray[index]
        self.teacherArray.remove(current)

    def Give_GenerationFromTeacher(self,num):
        count = np.shape(self.teacherArray)[0]
        t = int(num / 2)
        arr = []
        for i in range(t):
            index = random.randint(1,count) - 1
            new1 = self.teacherArray[index].GenerateImages(2)
            if np.shape(arr)[0] == 0:
                arr = new1
            else:
                arr = torch.cat([arr,new1],0)
        return arr

    def Give_GenerationFromTeacher_Cpu(self,num):
        count = np.shape(self.teacherArray)[0]
        t = int(num / 2)
        arr = []
        for i in range(t):
            index = random.randint(1,count) - 1
            new1 = self.teacherArray[index].GenerateImages(2)
            new1 = new1.unsqueeze(0).cuda().cpu()
            new1 = np.array(new1)
            new1 = new1[0]

            if np.shape(arr)[0] == 0:
                arr = new1
            else:
                arr = np.concatenate((arr,new1),0)
                #arr = torch.cat([arr,new1],0)
        return arr


    def Create_NewComponent(self):
        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        print(args.image_size)

        args.image_size = self.input_size
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())
        newTeacher = DiffusionComponent(model, diffusion,args,self.input_size,self.batch_size)

        '''
        if np.shape(self.teacherArray)[0] > 1:
            current = self.teacherArray[np.shape(self.teacherArray)[0]-1]
            current.memoryBuffer = current.GenerateImages(self.batch_size)
        '''
        self.teacherArray.append(newTeacher)
        return newTeacher

    def Create_NewComponent_Cpu(self):
        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        print(args.image_size)

        args.image_size = self.input_size
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())
        newTeacher = DiffusionComponent(model, diffusion,args,self.input_size,self.batch_size)

        '''
        if np.shape(self.teacherArray)[0] > 1:
            current = self.teacherArray[np.shape(self.teacherArray)[0]-1]
            memoryBuffer = current.GenerateImages(self.batch_size)
            
            memoryBuffer = memoryBuffer.unsqueeze(0).cuda().cpu()
            memoryBuffer = np.array(memoryBuffer)
            memoryBuffer = memoryBuffer[0]
            current.memoryBuffer = memoryBuffer
        '''
        self.teacherArray.append(newTeacher)
        return newTeacher


    def KnowledgeTransferForStudent(self,epoch,memoryBuffer):
        minbatch = int(self.batch_size / np.shape(self.teacherArray)[0])
        count = int(np.shape(memoryBuffer)[0]/minbatch)
        for i in range(epoch):
            n_examples = np.shape(memoryBuffer)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            memoryBuffer = memoryBuffer[index2]

            for j in range(count):
                batch = memoryBuffer[j*minbatch:(j+1)*minbatch]
                for c in range(np.shape(self.teacherArray)[0]):
                    gen = self.teacherArray[c].GenerateImages(minbatch)
                    batch = torch.cat([batch,gen],0)
                self.student.Train_One(batch)

    def KnowledgeTransferForStudent_Cpu(self,epoch,memoryBuffer):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        minbatch = int(self.batch_size / np.shape(self.teacherArray)[0])
        count = int(np.shape(memoryBuffer)[0]/minbatch)
        for i in range(epoch):
            n_examples = np.shape(memoryBuffer)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            memoryBuffer = memoryBuffer[index2]

            for j in range(count):
                batch = memoryBuffer[j*minbatch:(j+1)*minbatch]
                batch = torch.tensor(batch).cuda().to(device=device, dtype=torch.float)

                for c in range(np.shape(self.teacherArray)[0]):
                    gen = self.teacherArray[c].GenerateImages(minbatch)
                    batch = torch.cat([batch,gen],0)
                self.student.Train_One(batch)


    def KnowledgeTransferForStudent2(self,epoch):
        minbatch = int(self.batch_size / np.shape(self.teacherArray)[0])
        count = 10
        for i in range(epoch):
            for j in range(count):
                batch = []
                for c in range(np.shape(self.teacherArray)[0]):
                    gen = self.teacherArray[c].GenerateImages(minbatch)

                    if np.shape(batch)[0] == 0:
                        batch = gen
                    else:
                        batch = torch.cat([batch,gen],0)
                self.student.Train_One(batch)

    def _extract_into_tensor(self,arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def Calculate_JS(self,TSFramework, batch, batchReco):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        miniBatch = 64

        batch = batch.reshape(np.shape(batch)[0], self.input_size * self.input_size * 3)
        batchReco = batchReco.reshape(np.shape(batchReco)[0], self.input_size * self.input_size * 3)
        std = np.zeros((np.shape(batch)))
        std[:, :] = 0.01
        std = torch.tensor(std).cuda().to(device=device, dtype=torch.float)

        t = 100
        diffusion = TSFramework.teacherArray[0].diffusion
        schedule_sampler = UniformSampler(diffusion)
        times, weights = schedule_sampler.sample(np.shape(batch)[0], dist_util.dev())
        for i in range(np.shape(times)[0]):
            times[i] = t

        beta = self._extract_into_tensor(TSFramework.teacherArray[0].diffusion.sqrt_alphas_cumprod, times, batch.shape)

        batch = batch * beta
        batchReco = batchReco * beta

        q_z1 = td.normal.Normal(batch, std)
        q_z2 = td.normal.Normal(batchReco, std)
        score11 = td.kl_divergence(q_z1, q_z2).mean()
        score12 = td.kl_divergence(q_z2, q_z1).mean()

        score11 = score11 / miniBatch
        score12 = score12 / miniBatch
        score = (score11 + score12) / 2.0
        return score

    def Calculate_Relationship(self,component1,component2):
        #memory1 = component1.GenerateImages(128) #component1.memoryBuffer
        #memory2 = component1.GenerateImages(128)#component2.memoryBuffer

        memory1 = component1.memoryBuffer
        memory2 = component2.memoryBuffer

        distance = self.Calculate_JS(self,memory1,memory2)
        return distance

    def Calculate_RelationshipAll(self):
        componentArr1 = []
        componentArr2 = []
        arr = []
        ComponentCount = np.shape(self.teacherArray)[0]

        isState = False
        if np.shape(self.resultMatrix)[0] == 0:
            isState = True
            self.resultMatrix = np.zeros((ComponentCount,ComponentCount))

        for i in range(ComponentCount):
            for j in range(i + 1, ComponentCount):
                pattern = self.teacherArray[i]
                child = self.teacherArray[j]
                distance = self.Calculate_Relationship(pattern, child)
                distance = distance.unsqueeze(0).cuda().cpu()
                distance = distance.numpy()
                distance = distance[0]
                arr.append(distance)
                componentArr1.append(i)
                componentArr2.append(j)

                if isState == True:
                    self.resultMatrix[i,j] = distance
                    self.resultMatrix[j, i] = distance

        return arr,componentArr1,componentArr2

    def RemoveComponents(self,n):
        #Step 1 update
        batchsize = 128
        for i in range(np.shape(self.teacherArray)[0]):
            aa = self.teacherArray[i].GenerateImages(batchsize)
            self.teacherArray[i].memoryBuffer = aa

        #Step 2 Calculate the relationship
        t = np.shape(self.teacherArray)[0] - n

        for i in range(t):
            arr, componentArr1, componentArr2 = self.Calculate_RelationshipAll()
            index1 = np.argsort(arr)
            componentArr1 = componentArr1[index1]
            componentArr2 = componentArr2[index1]
            arr = arr[index1]

            componentIndex1 = componentArr1[0]
            componentIndex2 = componentArr2[0]
            self.teacherArray.remove(self.teacherArray[componentIndex1])

    def Give_DiversityScore(self,index):
        componentCount = np.shape(self.teacherArray)[0]
        arr = []
        for i in range(componentCount):
            if i != index:
                current = self.teacherArray[index]
                other = self.teacherArray[i]
                score = self.Calculate_Relationship(current,other)
                score = score.unsqueeze(0).cuda().cpu()
                score = score.numpy()
                score = score[0]
                arr.append(score)

        meanScore = np.mean(arr)
        return meanScore

    def RemoveComponentsThreshold(self,threshold):
        #Step 1 update
        batchsize = 128
        for i in range(np.shape(self.teacherArray)[0]):
            aa = self.teacherArray[i].GenerateImages(batchsize)
            self.teacherArray[i].memoryBuffer = aa

        #Step 2 Calculate the relationship
        t = np.shape(self.teacherArray)[0]

        for i in range(t):
            #if np.shape(self.teacherArray)[0] <= 5:
            #    return

            arr, componentArr1, componentArr2 = self.Calculate_RelationshipAll()
            index1 = np.argsort(arr)

            arr = np.array(arr)
            componentArr1 = np.array(componentArr1)
            componentArr2 = np.array(componentArr2)

            componentArr1 = componentArr1[index1]
            componentArr2 = componentArr2[index1]
            arr = arr[index1]

            v = arr[0]
            if v > threshold:
                return

            componentIndex1 = componentArr1[0]
            componentIndex2 = componentArr2[0]
            score1 = self.Give_DiversityScore(componentIndex1)
            score2 = self.Give_DiversityScore(componentIndex2)

            removiedIndex = componentIndex2
            if score1 > score2:
                removiedIndex = componentIndex1

            self.teacherArray.remove(self.teacherArray[removiedIndex])

class MemoryUnitFrameworkSupervised(MemoryUnitFramework):
    def __init__(self,name,device,input_size,dim):
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
        self.dim = dim

        '''
        if input_size == 256:
            self.student = Autoencoder256()
        else:
            self.student = Autoencoder(device,input_size)
        '''

        self.student = Autoencoder(device, input_size)
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
        self.expansionThreshold = 0.02
        self.memoryUnitSize = 64

        self.currentMemoryY = []
        self.memoryUnitsY = []
        self.outputSize = 10

    def GiveMemorizedSamples(self):
        if self.unitCount == 0:
            return self.currentMemory,self.currentMemoryY
        else:
            memory = self.currentMemory
            memoryY = self.currentMemoryY
            for i in range(self.unitCount):
                old = self.memoryUnits[i]
                oldY = self.memoryUnitsY[i]
                memory = th.cat([memory,old],0)
                memoryY = th.cat([memoryY,oldY],0)
            return memory,memoryY


    def RemoveData_FromMemory(self):
        arr = []
        nodeArr1 = []
        nodeArr2 = []
        for i in range(self.unitCount):
            for j in range(i+1,self.unitCount):
                nodeArr1.append(i)
                nodeArr2.append(j)
                old = self.memoryUnits[i]
                other = self.memoryUnits[j]
                mmd = self.Calculate_MMD(old, other)
                arr.append(mmd)

        index = np.argmin(arr)
        removedIndex1 = nodeArr1[index]
        removedIndex2 = nodeArr2[index]
        score1 = self.Calculate_DiscrepancyScore(removedIndex1)
        score2 = self.Calculate_DiscrepancyScore(removedIndex2)

        removedIndex = removedIndex1
        if score1 > score2:
            removedIndex = removedIndex2

        removedNode = self.memoryUnits[removedIndex]

        newMemoryArr = []
        newMemoryArrY = []

        for h in range(self.unitCount):
            if h != removedIndex:
                newMemoryArr.append(self.memoryUnits[h])
                newMemoryArrY.append(self.memoryUnitsY[h])

        #self.memoryUnits.remove(removedNode)
        self.memoryUnits = newMemoryArr
        self.memoryUnitsY = newMemoryArrY

        self.unitCount = self.unitCount-1

    def Check_Memory_Expansion(self):
        state = False
        measure = 0
        if self.unitCount == 0:
            self.memoryUnits.append(self.currentMemory)
            self.memoryUnitsY.append(self.currentMemoryY)

            self.currentMemory = []
            self.currentMemoryY = []
            self.unitCount += 1
        else:
            arr = []
            #Check the memory expansion
            for i in range(self.unitCount):
                old = self.memoryUnits[i]
                mmd = self.Calculate_MMD(old,self.currentMemory)
                arr.append(mmd)

            minV = np.min(arr)
            measure = minV
            if minV > self.threshold:
                self.memoryUnits.append(self.currentMemory)
                self.memoryUnitsY.append(self.currentMemoryY)

                self.unitCount += 1
                self.currentMemory = []
                self.currentMemoryY = []
                state = True

        #Check the maximum size
        if self.GiveMemorySize() > self.maxMemorySize:
            self.RemoveData_FromMemory()

        return measure,state

    def SampleSelection2(self,dataBatch,dataBatchY):
        if np.shape(self.currentMemory)[0] == 0:
            return dataBatch,dataBatchY

        memory = self.currentMemory
        memory2 = dataBatch

        memoryY = self.currentMemoryY
        memory2Y = dataBatchY

        #print(np.shape(memory))
        #print(np.shape(memory2))

        if np.shape(self.autoencoderArr)[0] > 1:

            lossArr = []
            lossArr2 = []

            for i in range(np.shape(self.autoencoderArr)[0]):
                a1 = self.autoencoderArr[i].vae.GiveLoss(self.currentMemory)
                a2 = self.autoencoderArr[i].vae.GiveLoss(memory2)

                a1 = np.array(a1)
                a2 = np.array(a2)

                if np.shape(lossArr)[0] == 0:
                    lossArr = a1
                    lossArr2 = a2
                else:
                    lossArr = a1 + lossArr
                    lossArr2 = a2 + lossArr2
        else:
            lossArr = self.student.vae.GiveLoss(self.currentMemory)
            lossArr2 = self.student.vae.GiveLoss(memory2)

        #print(np.shape(lossArr))
        #print(lossArr)

        memory = th.cat([memory,memory2],0)
        memoryY = th.cat([memoryY,memory2Y],0)

        lossArr = np.concatenate((lossArr,lossArr2),0)
        lossArr = lossArr*-1

        index = np.argsort(lossArr)

        memory = memory[index]
        memoryY = memoryY[index]

        memory = memory[0:self.memoryUnitSize]
        memoryY = memoryY[0:self.memoryUnitSize]

        return memory,memoryY

    def Create_NewComponent(self,output):
        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        print(args.image_size)


        args.image_size = self.input_size
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())

        newTeacher = DiffusionComponent(model, diffusion,args,self.input_size,self.batch_size)

        #Create Classifier

        if self.dim == 100:
            classifier = ResNet18_100(100).to(self.device)
        else:
            classifier = ResNet18().to(self.device)
        newTeacher.classifier = classifier

        '''
        if np.shape(self.teacherArray)[0] > 1:
            current = self.teacherArray[np.shape(self.teacherArray)[0]-1]
            current.memoryBuffer = current.GenerateImages(self.batch_size)
        '''
        self.teacherArray.append(newTeacher)
        return newTeacher

    def CheckExpansionByMMD(self,batch):
        isState = False

        arr = []
        for j in range(np.shape(self.teacherArray)[0]):
            generated = self.teacherArray[j].GenerateImages(64)
            mmd = self.Calculate_MMD(batch,generated)
            arr.append(mmd)

        score = np.min(arr)
        print("Expansion")
        print(score)
        if score > self.expansionThreshold:
            print("Build")
            newComponent = self.Create_NewComponent(self.outputSize)
            self.currentComponent = newComponent
            a1 = Autoencoder(self.device, self.input_size)
            self.autoencoderArr.append(a1)

            self.memoryUnits.clear()
            self.memoryUnitsY.clear()
            self.memoryUnits = []
            self.memoryUnitsY = []
            self.unitCount = 0
            isState = True

        return isState

class DynamicDiffusionMixture256(nn.Module):
    def __init__(self, name, device, input_size):
        super(DynamicDiffusionMixture256, self).__init__()

        self.input_size = input_size
        self.device = device
        self.trainingCount = 0
        self.trainingUpdate = 4
        self.GeneratingBatchSampleSize = 64
        self.batchTrainStudent_size = 64
        self.isTrainer = 0
        self.teacherArray = []
        self.student = VAE()
        self.currentComponent = 0
        self.batch_size = 64

        self.resultMatrix = []

    def create_argparser(self):
        defaults = dict(
            data_dir="/scratch/fy689/improved-diffusion-main/cifar_train",
            schedule_sampler="uniform",
            lr=1e-4,
            weight_decay=0.0,
            lr_anneal_steps=0,
            batch_size=128,
            microbatch=-1,  # -1 disables microbatches
            ema_rate="0.9999",  # comma-separated list of EMA values
            log_interval=10,
            save_interval=10000,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

    def TrainStudent(self, epoch, memory):
        # using the KD
        self.student.Train_Self_Single_Beta3(epoch, memory)

    def TrainStudent_Numpy(self,epoch,memory):
        self.student.Train_Self_Single_Beta3_Numpy(epoch,memory)

    def TrainStudent_Cpu(self, epoch, memory):
        # using the KD
        self.student.Train_Self_Single_Beta3_Cpu(epoch, memory)
        iterations = int(np.shape(memory)[0] / self.batch_size)
        arr = memory
        for s in range(epoch):
            n_examples = np.shape(arr)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            arr = arr[index2]

            for i in range(iterations):
                batch = arr[i*self.batch_size:(i+1)*self.batch_size]
                batch = torch.tensor(batch).cuda().to(device=self.device, dtype=th.float)
                self.student.Update(batch)


    def TrainStudent_Cpu_WithBeta(self, epoch, memory, beta):
        # using the KD
        self.student.Train_Self_Single_Beta3_Cpu_WithBeta(epoch, memory, beta)

    def RemoveExpertFromindex(self, index):
        current = self.teacherArray[index]
        self.teacherArray.remove(current)

    def Give_GenerationFromTeacher(self, num):
        count = np.shape(self.teacherArray)[0]
        t = int(num / 2)
        arr = []
        for i in range(t):
            index = random.randint(1, count) - 1
            new1 = self.teacherArray[index].GenerateImages(2)
            if np.shape(arr)[0] == 0:
                arr = new1
            else:
                arr = torch.cat([arr, new1], 0)
        return arr

    def Create_NewComponent(self):
        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        print(args.image_size)

        args.image_size = self.input_size
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())
        newTeacher = DiffusionComponent(model, diffusion, args, self.input_size, self.batch_size)

        '''
        if np.shape(self.teacherArray)[0] > 1:
            current = self.teacherArray[np.shape(self.teacherArray)[0]-1]
            current.memoryBuffer = current.GenerateImages(self.batch_size)
        '''
        self.teacherArray.append(newTeacher)
        return newTeacher

    def Create_NewComponent_Cpu(self):
        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        print(args.image_size)

        args.image_size = self.input_size
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())
        newTeacher = DiffusionComponent(model, diffusion, args, self.input_size, self.batch_size)

        '''
        if np.shape(self.teacherArray)[0] > 1:
            current = self.teacherArray[np.shape(self.teacherArray)[0]-1]
            memoryBuffer = current.GenerateImages(self.batch_size)

            memoryBuffer = memoryBuffer.unsqueeze(0).cuda().cpu()
            memoryBuffer = np.array(memoryBuffer)
            memoryBuffer = memoryBuffer[0]
            current.memoryBuffer = memoryBuffer
        '''
        self.teacherArray.append(newTeacher)
        return newTeacher

    def KnowledgeTransferForStudent(self, epoch, memoryBuffer):
        minbatch = int(self.batch_size / np.shape(self.teacherArray)[0])
        count = int(np.shape(memoryBuffer)[0] / minbatch)
        for i in range(epoch):
            n_examples = np.shape(memoryBuffer)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            memoryBuffer = memoryBuffer[index2]

            for j in range(count):
                batch = memoryBuffer[j * minbatch:(j + 1) * minbatch]
                for c in range(np.shape(self.teacherArray)[0]):
                    gen = self.teacherArray[c].GenerateImages(minbatch)
                    batch = torch.cat([batch, gen], 0)
                self.student.Train_One(batch)

    def KnowledgeTransferForStudent_Cpu(self, epoch, memoryBuffer):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        minbatch = int(self.batch_size / np.shape(self.teacherArray)[0])
        count = int(np.shape(memoryBuffer)[0] / minbatch)
        for i in range(epoch):
            n_examples = np.shape(memoryBuffer)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            memoryBuffer = memoryBuffer[index2]

            for j in range(count):
                batch = memoryBuffer[j * minbatch:(j + 1) * minbatch]
                batch = torch.tensor(batch).cuda().to(device=device, dtype=torch.float)

                for c in range(np.shape(self.teacherArray)[0]):
                    gen = self.teacherArray[c].GenerateImages(minbatch)
                    batch = torch.cat([batch, gen], 0)
                self.student.Train_One(batch)

    def KnowledgeTransferForStudent2(self, epoch):
        minbatch = int(self.batch_size / np.shape(self.teacherArray)[0])
        count = 10
        for i in range(epoch):
            for j in range(count):
                batch = []
                for c in range(np.shape(self.teacherArray)[0]):
                    gen = self.teacherArray[c].GenerateImages(minbatch)

                    if np.shape(batch)[0] == 0:
                        batch = gen
                    else:
                        batch = torch.cat([batch, gen], 0)
                self.student.Train_One(batch)

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def Calculate_JS(self, TSFramework, batch, batchReco):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        miniBatch = 64

        batch = batch.reshape(np.shape(batch)[0], self.input_size * self.input_size * 3)
        batchReco = batchReco.reshape(np.shape(batchReco)[0], self.input_size * self.input_size * 3)
        std = np.zeros((np.shape(batch)))
        std[:, :] = 0.01
        std = torch.tensor(std).cuda().to(device=device, dtype=torch.float)

        t = 100
        diffusion = TSFramework.teacherArray[0].diffusion
        schedule_sampler = UniformSampler(diffusion)
        times, weights = schedule_sampler.sample(np.shape(batch)[0], dist_util.dev())
        for i in range(np.shape(times)[0]):
            times[i] = t

        beta = self._extract_into_tensor(TSFramework.teacherArray[0].diffusion.sqrt_alphas_cumprod, times, batch.shape)

        batch = batch * beta
        batchReco = batchReco * beta

        q_z1 = td.normal.Normal(batch, std)
        q_z2 = td.normal.Normal(batchReco, std)
        score11 = td.kl_divergence(q_z1, q_z2).mean()
        score12 = td.kl_divergence(q_z2, q_z1).mean()

        score11 = score11 / miniBatch
        score12 = score12 / miniBatch
        score = (score11 + score12) / 2.0
        return score

    def Calculate_Relationship(self, component1, component2):
        # memory1 = component1.GenerateImages(128) #component1.memoryBuffer
        # memory2 = component1.GenerateImages(128)#component2.memoryBuffer

        memory1 = component1.memoryBuffer
        memory2 = component2.memoryBuffer

        distance = self.Calculate_JS(self, memory1, memory2)
        return distance

    def Calculate_RelationshipAll(self):
        componentArr1 = []
        componentArr2 = []
        arr = []
        ComponentCount = np.shape(self.teacherArray)[0]

        isState = False
        if np.shape(self.resultMatrix)[0] == 0:
            isState = True
            self.resultMatrix = np.zeros((ComponentCount, ComponentCount))

        for i in range(ComponentCount):
            for j in range(i + 1, ComponentCount):
                pattern = self.teacherArray[i]
                child = self.teacherArray[j]
                distance = self.Calculate_Relationship(pattern, child)
                distance = distance.unsqueeze(0).cuda().cpu()
                distance = distance.numpy()
                distance = distance[0]
                arr.append(distance)
                componentArr1.append(i)
                componentArr2.append(j)

                if isState == True:
                    self.resultMatrix[i, j] = distance
                    self.resultMatrix[j, i] = distance

        return arr, componentArr1, componentArr2

    def RemoveComponents(self, n):
        # Step 1 update
        batchsize = 128
        for i in range(np.shape(self.teacherArray)[0]):
            aa = self.teacherArray[i].GenerateImages(batchsize)
            self.teacherArray[i].memoryBuffer = aa

        # Step 2 Calculate the relationship
        t = np.shape(self.teacherArray)[0] - n

        for i in range(t):
            arr, componentArr1, componentArr2 = self.Calculate_RelationshipAll()
            index1 = np.argsort(arr)
            componentArr1 = componentArr1[index1]
            componentArr2 = componentArr2[index1]
            arr = arr[index1]

            componentIndex1 = componentArr1[0]
            componentIndex2 = componentArr2[0]
            self.teacherArray.remove(self.teacherArray[componentIndex1])

    def Give_DiversityScore(self, index):
        componentCount = np.shape(self.teacherArray)[0]
        arr = []
        for i in range(componentCount):
            if i != index:
                current = self.teacherArray[index]
                other = self.teacherArray[i]
                score = self.Calculate_Relationship(current, other)
                score = score.unsqueeze(0).cuda().cpu()
                score = score.numpy()
                score = score[0]
                arr.append(score)

        meanScore = np.mean(arr)
        return meanScore

    def RemoveComponentsThreshold(self, threshold):
        # Step 1 update
        batchsize = 128
        for i in range(np.shape(self.teacherArray)[0]):
            aa = self.teacherArray[i].GenerateImages(batchsize)
            self.teacherArray[i].memoryBuffer = aa

        # Step 2 Calculate the relationship
        t = np.shape(self.teacherArray)[0]

        for i in range(t):
            # if np.shape(self.teacherArray)[0] <= 5:
            #    return

            arr, componentArr1, componentArr2 = self.Calculate_RelationshipAll()
            index1 = np.argsort(arr)

            arr = np.array(arr)
            componentArr1 = np.array(componentArr1)
            componentArr2 = np.array(componentArr2)

            componentArr1 = componentArr1[index1]
            componentArr2 = componentArr2[index1]
            arr = arr[index1]

            v = arr[0]
            if v > threshold:
                return

            componentIndex1 = componentArr1[0]
            componentIndex2 = componentArr2[0]
            score1 = self.Give_DiversityScore(componentIndex1)
            score2 = self.Give_DiversityScore(componentIndex2)

            removiedIndex = componentIndex2
            if score1 > score2:
                removiedIndex = componentIndex1

            self.teacherArray.remove(self.teacherArray[removiedIndex])


