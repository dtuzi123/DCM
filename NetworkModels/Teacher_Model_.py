import argparse
import torchvision.transforms as transforms
import torch
from datasets.MyCIFAR10 import *

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.train_util_balance import TrainLoop_Balance
import torch.nn as nn

class Teacher(nn.Module):
    def __init__(self,inputSize):
        super(Teacher, self).__init__()

        print("build the teacher model")
        self.currentData_X = []
        self.currentData_Y = []
        self.input_size = inputSize
        self.batch_size = 64

        print(self.input_size)

        args = self.create_argparser().parse_args()
        self.args = args

        self.device = 0

        dist_util.setup_dist()
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

    def Sampling_By_NumAndNoise(self,diffusion,model,num,noise1):
        model_kwargs = None
        use_ddim = True
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (num, 3, self.input_size, self.input_size),
            clip_denoised=True,
            noise = noise1,
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

    def Give_Reconstruction_All(self,trainingData):
        batchSize = 64
        count = int(np.shape(trainingData)[0]/batchSize)

        recoArr = []
        for i in range(count):
            x = trainingData[i*batchSize:(i+1)*batchSize]
            reco = self.Give_Reconstruction_DataBatch(x)
            if np.shape(recoArr)[0] == 0:
                recoArr = reco
            else:
                recoArr = torch.cat([recoArr,reco],0)
        return recoArr

    def Give_Reconstruction_DataBatch(self,dataBatch):

        mytimes = []
        for i in range(np.shape(dataBatch)[0]):
            mytimes.append(1000)

        mytimes = torch.tensor(mytimes).cuda().to(device=self.device, dtype=torch.long)

        args = self.args
        #schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, self.diffusion)

        import torch as th
        batch_size = 64
        aa = np.ones([self.diffusion.num_timesteps])
        w = aa
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(self.device)
        mytimes = indices

        latexntX = self.diffusion.q_sample(dataBatch,mytimes)
        count = np.shape(latexntX)[0]
        reconstructions = self.Sampling_By_NumAndNoise(self.diffusion,self.model,count,latexntX)
        return reconstructions

    def Give_Generation(self,n):
        batchSize = self.batch_size
        if n < batchSize:
            return self.Sampling_By_Num(self.diffusion, self.model, n)
        else:
            count = int(n / batchSize)
            arr = []
            for j in range(count):
                a1 = self.Sampling_By_Num(self.diffusion, self.model, batchSize)
                if np.shape(arr)[0] == 0:
                    arr = a1
                else:
                    arr = torch.cat([arr, a1], dim=0)
        return arr

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


    def Train_Self_ByDataLoad_Single(self,epoch,dataX):
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
        ).train_self_bySingleData(epoch,self.device, dataX)
        self.trainer = trainer


    def Train_Self(self,epoch,mydata):
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

        trainer = TrainLoop(
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
        ).train_self(epoch, mydata)
        self.trainer = trainer

class Balance_Teacher(Teacher):

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
