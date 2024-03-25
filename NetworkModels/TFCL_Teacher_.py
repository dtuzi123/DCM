import argparse
import torchvision.transforms as transforms
import torch
from improved_diffusion.resample import *
from datasets.MyCIFAR10 import *

from improved_diffusion import *
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from NetworkModels.Teacher_Model_ import *
from improved_diffusion.train_util_balance_NoMPI_MultiGPU import *

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.train_util_balance import TrainLoop_Balance
import torch.nn as nn

class TFCL_SubTeacher(nn.Module):
    def __init__(self,model,diffusion):
        super(TFCL_SubTeacher, self).__init__()

        #super(TFCL_SubTeacher, self).__init__()
        self.model = model
        self.diffusion = diffusion
        self.currentKnowledge = []

    def GenerateImages(self,num):
        samples = self.model.Sampling_By_Num(self.diffusion, self.model, num)
        return samples

class TFCL_Teacher(nn.Module):
    def __init__(self,inputSize):
        super(TFCL_Teacher, self).__init__()

        print("build the teacher model")
        self.currentData_X = []
        self.currentData_Y = []
        self.input_size = inputSize
        self.batch_size = 64

        self.trainer = 0
        self.isTrainer = 0

        print(self.input_size)

        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        # dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        print(args.image_size)

        args.image_size = inputSize
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        self.model = model
        self.diffusion = diffusion

        newTeacher = TFCL_SubTeacher(model,diffusion)
        self.teacherArray = []
        self.teacherArray.append(newTeacher)
        self.currentTeacher = newTeacher


    def Give_Reconstruction_All(self,trainingData,device,model,diffusion):
        batchSize = self.batch_size
        count = int(np.shape(trainingData)[0]/batchSize)

        recoArr = []
        for i in range(count):
            x = trainingData[i*batchSize:(i+1)*batchSize]
            reco = self.Give_Reconstruction_DataBatch(x,device,model,diffusion)
            if np.shape(recoArr)[0] == 0:
                recoArr = reco
            else:
                recoArr = torch.cat([recoArr,reco],0)
        return recoArr


    def Give_Reconstruction_DataBatch(self, dataBatch, device,model,diffusion):

        mytimes = []
        for i in range(np.shape(dataBatch)[0]):
            mytimes.append(1000)

        mytimes = torch.tensor(mytimes).cuda().to(device=device, dtype=torch.long)

        import torch as th
        batch_size = self.batch_size
        aa = np.ones([self.diffusion.num_timesteps])
        w = aa
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        mytimes = indices

        latexntX = self.diffusion.q_sample(dataBatch, mytimes)
        count = np.shape(latexntX)[0]
        reconstructions = self.Sampling_By_NumAndNoise(diffusion, model, count, latexntX)
        return reconstructions

    def q_sample(self,x_start, t, noise=None):
        schedule_sampler = UniformSampler(self.diffusion)
        times, weights = schedule_sampler.sample(np.shape(x_start)[0], dist_util.dev())
        for i in range(np.shape(times)[0]):
            times[i] = t

        x_t = self.diffusion.q_sample(x_start, times, noise=noise)
        return x_t

    def Sampling_By_NumAndNoise(self, diffusion, model, num, noise1):
        model_kwargs = None
        use_ddim = True
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (num, 3, self.input_size, self.input_size),
            clip_denoised=True,
            noise=noise1,
            model_kwargs=model_kwargs,
        )
        '''
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)
        mySamples = mySamples[0]
        '''
        return sample

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

    def GetTeacherCount(self):
        return np.shape(self.teacherArray)[0]

    def Create_NewTeacher(self):
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
        newTeacher = TFCL_SubTeacher(model, diffusion)
        self.teacherArray.append(newTeacher)
        return newTeacher

    def Train_Self_ByDataLoad2(self,epoch,mydata):
        args = self.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=self.input_size,  # args.image_size,
            class_cond=args.class_cond,
        )
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size


        trainer = TrainLoop_Balance_NoMPI_MultiGPU(
            model=model,
            diffusion=diffusion,
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
        ).train_self_bySingleData_Unsupervised(epoch,self.device, mydata)
        self.trainer = trainer

    def Train_Self_ByDataLoad(self,epoch,mydata,generatedData):
        args = self.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=self.input_size,  # args.image_size,
            class_cond=args.class_cond,
        )
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        trainer = TrainLoop_Balance(
            model=model,
            diffusion=diffusion,
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
        ).train_self_byDataLoad(epoch,self.device, mydata,generatedData)
        self.trainer = trainer

    def Train_Self_(self,epoch,mydata,generatedData):
        args = self.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=self.input_size,  # args.image_size,
            class_cond=args.class_cond,
        )
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        trainer = TrainLoop_Balance(
            model=model,
            diffusion=diffusion,
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
        ).train_self(epoch, mydata,generatedData)
        self.trainer = trainer

    def GenerateImages(self,diffusion,model,num):

        if num < self.batch_size:
            new1 = self.Sampling_By_Num(diffusion,model,num)
            return new1

        count = int(num / self.batch_size)
        arr = []
        for i in range(count):
            new1 = self.Sampling_By_Num(diffusion,model,self.batch_size)
            if np.shape(arr)[0] == 0:
                arr = new1
            else:
                arr = torch.cat([arr,new1],0)
        return arr

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
        '''
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)
        mySamples = mySamples[0]
        '''
        return sample


    def SamplingNoiseImages(self,diffusion,model,num,t):
        model_kwargs = None
        use_ddim = True
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop_ByDefinedTime
        )

        sample,noiseImage = sample_fn(
            model,
            (num, 3, self.input_size, self.input_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
            time = t,
        )
        return sample,noiseImage

    def SaplingBy_Num_WithImages(self,diffusion,model,batch,t):

        schedule_sampler = UniformSampler(diffusion)
        times, weights = schedule_sampler.sample(np.shape(batch)[0], dist_util.dev())

        for i in range(np.shape(times)[0]):
            times[i] = t

        noiseBatch = diffusion.q_sample(batch, times, None)
        return batch,noiseBatch


    def SaplingHalf_By_Num_WithImages(self,diffusion,model,batch):

        schedule_sampler = UniformSampler(diffusion)
        times, weights = schedule_sampler.sample(np.shape(batch)[0], dist_util.dev())

        for i in range(np.shape(times)[0]):
            times[i] = 500

        noiseBatch = diffusion.q_sample(batch, times, None)
        return batch,noiseBatch

    def SaplingHalf_By_NumAndStep_WithImages(self,diffusion,model,batch,step):

        schedule_sampler = UniformSampler(diffusion)
        times, weights = schedule_sampler.sample(np.shape(batch)[0], dist_util.dev())

        for i in range(np.shape(times)[0]):
            times[i] = step

        noiseBatch = diffusion.q_sample(batch, times, None)
        return batch,noiseBatch


    def SamplingHalf_By_Num(self,diffusion,model,num):
        model_kwargs = None
        use_ddim = True
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop_Half
        )

        sample,noiseSample = sample_fn(
            model,
            (num, 3, self.input_size, self.input_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        '''
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)
        mySamples = mySamples[0]
        '''
        return sample,noiseSample


    def Sampling_By_NumAndStep(self,diffusion,model,num,step):
        model_kwargs = None
        use_ddim = True
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop_Step
        )

        sample,noiseSample = sample_fn(
            model,
            (num, 3, self.input_size, self.input_size),
            clip_denoised=True,
            step = step,
            model_kwargs=model_kwargs,
        )
        '''
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)
        mySamples = mySamples[0]
        '''
        return sample,noiseSample
