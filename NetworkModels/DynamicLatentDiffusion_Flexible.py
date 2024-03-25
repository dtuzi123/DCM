
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
from NetworkModels.MemoryUnitFramework_ import *
from sklearn.preprocessing import MinMaxScaler
from improved_diffusion.unet import *
from improved_diffusion.script_util import *
from models.FullyVAE_ import *

#
class LatentDiffusionComponent_Flexible(nn.Module):
    def __init__(self,model,diffusion,args,inputsize,batchsize,latentSize):
        super(LatentDiffusionComponent_Flexible, self).__init__()
        self.model = model
        self.diffusion = diffusion
        self.isTrainer = 0
        self.args = args
        self.input_size = inputsize
        self.batch_size = batchsize
        self.memoryBuffer = 0
        self.latentSize = latentSize

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.autoencoder = VanillaVAE_Flexible(in_channels = 3,latent_dim=latentSize*latentSize*3,input_size=self.input_size,beta=1)
        #self.autoencoder = FullVAE(in_channels = 3,latent_dim=latentSize*latentSize*3,input_size=self.input_size,beta=1)

        self.autoencoder.to(device)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)

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

    def Calculate_VAL(self,batch):
        r = self.teacher.diffusion.Calculate_VA( self.teacher.model,batch)
        return r

    def Transfer_To_Numpy(self,sample):
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        #sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)
        mySamples = mySamples[0]
        return mySamples

    def normalization(self,X):
        X = (X - X.min()) / (X.max() - X.min()) * 2 - 1
        return X

    def Train_Autoencoder(self,epoch,memory):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        memory = torch.tensor(memory).cuda().to(device=device, dtype=torch.float)

        count = int(np.shape(memory)[0] / self.batch_size)
        for i in range(epoch):
            for j in range(count):
                batch = memory[j*self.batch_size:(j+1)*self.batch_size]

                self.optimizer.zero_grad()
                results = self.autoencoder.forward(batch)
                train_loss = self.autoencoder.loss_function(*results)
                loss = train_loss['loss']
                loss.backward()
                self.optimizer.step()

    def Train_Small(self,epoch,memory):
        args = self.args

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.model
        currentDiffusion = self.diffusion

        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        latentMemory = self.autoencoder.GiveFeatures(memory)

        latentMemory = latentMemory.view(np.shape(latentMemory)[0], 3, self.latentSize, self.latentSize)

        #print(np.shape(latentMemory))
        #latentMemory = self.Transfer_To_Numpy(latentMemory)
        #latentMemory = self.normalization(latentMemory)

        #latentMemory = th.tensor(latentMemory).cuda().to(device=device, dtype=th.float)

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

            self.trainer.train_Memory(epoch, latentMemory)
        else:
            self.trainer.train_Memory(epoch, latentMemory)

    def Train(self,epoch,memory):
        args = self.args

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

            self.trainer.train_Memory(epoch, memory)
        else:
            self.trainer.train_Memory(epoch, memory)

    def Train_Cpu_WithFiles(self,epoch,memory):
        args = self.args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

            self.trainer.train_Memory_Cpu_WithFiles(epoch, memory,device)
        else:
            self.trainer.train_Memory_Cpu_WithFiles(epoch, memory,device)

    def Train_Cpu(self,epoch,memory):
        args = self.args
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        latentSize = self.latentSize

        model_kwargs = None
        use_ddim = True
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        latentSample = sample_fn(
            model,
            (num, 3, latentSize, latentSize),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )

        latentSample = latentSample.reshape(np.shape(latentSample)[0],-1)
        sample = self.autoencoder.Give_FullSampleByLatent(latentSample)

        return sample

    def q_sample(self,x_start, t, noise=None):
        schedule_sampler = UniformSampler(self.diffusion)
        times, weights = schedule_sampler.sample(np.shape(x_start)[0], dist_util.dev())
        for i in range(np.shape(times)[0]):
            times[i] = t

        x_t = self.diffusion.q_sample(x_start, times, noise=noise)
        return x_t

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


class DynamicLatentDiffusion_Flexible(MemoryUnitFramework):
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
        self.diffusionArr = []
        self.latentSize = 16

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


    def KnowledgeTransfer(self,epoch,memory):
        smallBatch = int(self.batch_size/2)
        myCount = int(np.shape(memory)[0] / smallBatch)
        for i in range(epoch):
            generatedBatch = self.diffusionArr[0].GenerateImages(np.shape(memory)[0])
            newData = torch.cat([memory,generatedBatch],0)
            self.teacherArray[0].Train(epoch,newData)

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

        newTeacher = LatentDiffusionComponent_Flexible(model, diffusion,args,self.input_size,self.batch_size,self.latentSize)

        '''
        if np.shape(self.teacherArray)[0] > 1:
            current = self.teacherArray[np.shape(self.teacherArray)[0]-1]
            current.memoryBuffer = current.GenerateImages(self.batch_size)
        '''
        self.diffusionArr.append(newTeacher)
        return newTeacher

    def CreateLatentDiffusion(self):
        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        print(args.image_size)

        args.image_size = self.latentSize
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())
        newTeacher = LatentDiffusionComponent_Flexible(model, diffusion,args,self.input_size,self.batch_size,self.latentSize)

        '''
        if np.shape(self.teacherArray)[0] > 1:
            current = self.teacherArray[np.shape(self.teacherArray)[0]-1]
            current.memoryBuffer = current.GenerateImages(self.batch_size)
        '''
        self.teacherArray.append(newTeacher)
        return newTeacher

    def Give_GenerationFromTeacher(self,num):

        t = int(num / self.batch_size)
        arr = []
        for i in range(t):
            new1 = self.teacherArray[0].GenerateImages(self.batch_size)
            if np.shape(arr)[0] == 0:
                arr = new1
            else:
                arr = torch.cat([arr,new1],0)
        return arr