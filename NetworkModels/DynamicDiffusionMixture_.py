
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
import torch.nn.functional as F
import torch.optim as optim

class DiffusionComponent(nn.Module):
    def __init__(self,model,diffusion,args,inputsize,batchsize):
        super(DiffusionComponent, self).__init__()
        self.model = model
        self.diffusion = diffusion
        self.isTrainer = 0
        self.args = args
        self.input_size = inputsize
        self.batch_size = batchsize
        self.memoryBuffer = 0
        self.classifier = 0
        self.classifierSGD = 0
        self.originalInputSize = 256

    def TrainClassifier_BatchSize(self,epoch,memoryX,memoryY,batchSize):
        criterion = nn.CrossEntropyLoss()
        LR = 0.01
        if self.classifierSGD == 0:
            optimizer = optim.SGD(self.classifier.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            self.classifierSGD = optimizer

        batchSize = batchSize
        count = int(np.shape(memoryX)[0] / batchSize)

        dataX = memoryX
        dataY = memoryY

        for j in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]
            dataY = dataY[index2]

            for i in range(count):
                batchX = dataX[i * batchSize:(i + 1) * batchSize]
                batchY = dataY[i * batchSize:(i + 1) * batchSize]

                self.classifierSGD.zero_grad()
                # forward & backward
                outputs = self.classifier(batchX)
                loss = criterion(outputs, batchY)
                loss.backward()
                self.classifierSGD.step()

        return loss.unsqueeze(0).cuda().cpu()

    def TrainClassifier(self,epoch,memoryX,memoryY):
        criterion = nn.CrossEntropyLoss()
        LR = 0.1
        if self.classifierSGD == 0:
            optimizer = optim.SGD(self.classifier.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            self.classifierSGD = optimizer

        batchSize = 64
        count = int(np.shape(memoryX)[0] / batchSize)

        dataX = memoryX
        dataY = memoryY

        for j in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]
            dataY = dataY[index2]

            for i in range(count):
                batchX = dataX[i * batchSize:(i + 1) * batchSize]
                batchY = dataY[i * batchSize:(i + 1) * batchSize]

                self.classifierSGD.zero_grad()

                # forward & backward
                outputs = self.classifier(batchX)
                loss = criterion(outputs, batchY)
                loss.backward()
                self.classifierSGD.step()
        return loss.unsqueeze(0).cuda().cpu()

    def Calculate_VAL(self,batch):
        r = self.teacher.diffusion.Calculate_VA( self.teacher.model,batch)
        return r

    def Train_Small(self,epoch,memory):
        args = self.args

        #print(self.input_size)

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.model
        currentDiffusion = self.diffusion

        if self.isTrainer == 0:
            self.isTrainer = 1
            self.trainer = TrainLoop_Balance_NoMPI_MultiGPU(
                model=currentTeacher,
                diffusion=currentDiffusion,
                data=data,
                batch_size=args.batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
            )  # .train_self(epoch, mydata,generatedData)

            self.trainer.train_Memory_Small(epoch, memory)
        else:
            self.trainer.train_Memory_Small(epoch, memory)

    def Train_Numpy(self,epoch,memory):
        args = self.args

        # print(self.input_size)

        # print("batchsize")
        # print(self.batch_size)

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = self.batch_size

        currentTeacher = self.model
        currentDiffusion = self.diffusion

        if self.isTrainer == 0:
            self.isTrainer = 1
            self.trainer = TrainLoop_Balance_NoMPI_MultiGPU(
                model=currentTeacher,
                diffusion=currentDiffusion,
                data=data,
                batch_size=args.batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
            )  # .train_self(epoch, mydata,generatedData)

            self.trainer.train_Memory_Numpy(epoch, memory)
        else:
            self.trainer.train_Memory_Numpy(epoch, memory)

    def Train(self,epoch,memory):
        args = self.args

        #print(self.input_size)

        #print("batchsize")
        #print(self.batch_size)

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = self.batch_size

        currentTeacher = self.model
        currentDiffusion = self.diffusion

        if self.isTrainer == 0:
            self.isTrainer = 1
            self.trainer = TrainLoop_Balance_NoMPI_MultiGPU(
                model=currentTeacher,
                diffusion=currentDiffusion,
                data=data,
                batch_size=args.batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
            )  # .train_self(epoch, mydata,generatedData)

            self.trainer.train_Memory(epoch, memory)
        else:
            self.trainer.train_Memory(epoch, memory)

    def Train_Cpu_WithFiles(self,epoch,memory):
        args = self.args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #print(self.input_size)

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.model
        currentDiffusion = self.diffusion

        if self.isTrainer == 0:
            self.isTrainer = 1
            self.trainer = TrainLoop_Balance_NoMPI_MultiGPU(
                model=currentTeacher,
                diffusion=currentDiffusion,
                data=data,
                batch_size=args.batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
            )  # .train_self(epoch, mydata,generatedData)

            self.trainer.train_Memory_Cpu_WithFiles(epoch, memory,device,self.input_size, self.originalInputSize)
        else:
            self.trainer.train_Memory_Cpu_WithFiles(epoch, memory,device,self.input_size,self.originalInputSize)

    def Train_Cpu_WithFilesAndSize(self, epoch, memory):
        args = self.args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # print(self.input_size)

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.model
        currentDiffusion = self.diffusion

        if self.isTrainer == 0:
            self.isTrainer = 1
            self.trainer = TrainLoop_Balance_NoMPI_MultiGPU(
                model=currentTeacher,
                diffusion=currentDiffusion,
                data=data,
                batch_size=args.batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
            )  # .train_self(epoch, mydata,generatedData)

            self.trainer.train_Memory_Cpu_WithFilesAndSize(epoch, memory, device,self.input_size)
        else:
            self.trainer.train_Memory_Cpu_WithFilesAndSize(epoch, memory, device,self.input_size)

    def Train_Cpu(self,epoch,memory):
        args = self.args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #print(self.input_size)

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.model
        currentDiffusion = self.diffusion

        if self.isTrainer == 0:
            self.isTrainer = 1
            self.trainer = TrainLoop_Balance_NoMPI_MultiGPU(
                model=currentTeacher,
                diffusion=currentDiffusion,
                data=data,
                batch_size=args.batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
            )  # .train_self(epoch, mydata,generatedData)

            self.trainer.train_Memory_Cpu(epoch, memory,device)
        else:
            self.trainer.train_Memory_Cpu(epoch, memory,device)

    def Sampling_By_Num(self,diffusion,model,num):
        model_kwargs = None
        use_ddim = True
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (num, 3, self.input_size, self.input_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        return sample

    def q_sample(self,x_start, t, noise=None):
        schedule_sampler = UniformSampler(self.diffusion)
        times, weights = schedule_sampler.sample(np.shape(x_start)[0], dist_util.dev())
        for i in range(np.shape(times)[0]):
            times[i] = t

        x_t = self.diffusion.q_sample(x_start, times, noise=noise)
        return x_t

    def Transfer_To_Numpy(self,sample):
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        #sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)
        mySamples = mySamples[0]
        return mySamples

    def GenerateImage_CPU(self,num):
        batchsize = 64
        count = int(num / batchsize)

        arr = []
        for i in range(count):
            samples = self.Sampling_By_Num(self.diffusion, self.model, batchsize)
            samples = self.Transfer_To_Numpy(samples)
            if np.shape(arr)[0] == 0:
                arr = samples
            else:
                arr = np.concatenate((arr,samples),0)

        return arr

    def GenerateImages(self,num):
        samples = self.Sampling_By_Num(self.diffusion, self.model, num)
        return samples

    def GenerateImagesBig(self,num):
        batch = 64
        count = int(num / batch)
        result = []
        for i in range(count):
            samples = self.Sampling_By_Num(self.diffusion, self.model, batch)
            if np.shape(result)[0] == 0:
                result = samples
            else:
                result = th.cat([result,samples],0)
        return result

class DynamicDiffusionMixture(nn.Module):
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
        self.student = TFCL_StudentModel(device,input_size)
        self.currentComponent = 0
        self.batch_size = 64
        self.originalInputSize = 256

        self.resultMatrix = []

        self.maxTrainingStep = 100
        self.currentTrainingTime = 0
        self.currentMemory = []

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
        self.student.Train_Self_Single_Beta3(epoch,memory)

    def TrainingStudentFromTeacher(self,epoch,memory,isMemory):
        iterations = int(np.shape(memory)[0] / self.batch_size)

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
                    optimizer.step()


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

    def Create_NewComponent(self):

        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        #print(args.image_size)

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
        newTeacher.originalInputSize=self.originalInputSize
        newTeacher.input_size = self.input_size

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
            #print(v)
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
            #print(v)
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


