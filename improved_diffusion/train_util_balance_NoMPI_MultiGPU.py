import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import cv2
import os
from skimage import io, transform
from cv2_imageProcess import *
import torch as mytorch
import torch.nn as nn

#
from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

#new add
from improved_diffusion import dist_util, logger


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import argparse

class TrainLoop_Balance_NoMPI_MultiGPU:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.originalInputSize = 256

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size #* dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.device = device

        self.inputImageSize = 256

        #self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        #if th.cuda.device_count() > 1:
        #    self.model = nn.DataParallel(self.model)
        #    print("Multiple GPUs")
            #model.to(device)
        '''
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        '''
        self.use_ddp = False
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        if th.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
            print(f"Let's use {th.cuda.device_count()} GPUs!")
            #self.model = nn.DataParallel(self.model,device_ids=[0,1])  # 将模型对象转变为多GPU并行运算的模型
            self.use_ddp = True
            '''
            self.model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            '''

        self.model.to(device)
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()



    def Sampling_By_Num(self,diffusion,model,num):
        model_kwargs = None
        use_ddim = True
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (num, 3, 32, 32),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)
        mySamples = mySamples[0]
        return mySamples

    def Generate_Images(self,step):
        all_images = []
        all_labels = []
        class_cond = False
        batch_size = 64
        num_samples = 64

        model_kwargs = {}
        model_kwargs = None
        if False:
            NUM_CLASSES = 10
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(64,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        use_ddim = True
        sample_fn = (
            self.diffusion.p_sample_loop if not use_ddim else self.diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            self.model,
            (10, 3, 32, 32),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        mySamples = sample.unsqueeze(0).cuda().cpu()
        mySamples = np.array(mySamples)

        mySamples = mySamples[0]

        out1 = merge2(mySamples[:10], [1, 10])
        #print(np.shape(out1))

        path = "results/"
        #cv2.imwrite(os.path.join(path, 'waka.jpg'), mySamples)
        name = "CIFAR10_generated_" + str(step) + ".png"
        cv2.imwrite("/scratch/fy689/improved-diffusion-main/results/"+name, out1)
        cv2.waitKey(0)
        #io.imsave("/scratch/fy689/improved-diffusion-main/results/aa.png", mySamples[0])

    def train_self_bySingleData(self,epoch,device,dataX):
        batchSize = self.batch_size
        smallBatchSize = int(batchSize/2.0)
        count = int(np.shape(dataX)[0] / smallBatchSize)
        myCount = int(np.shape(generatedX)[0] / smallBatchSize)

        for i in range(epoch):

            '''
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            n_examples3 = np.shape(generatedX)[0]
            index3 = [i for i in range(n_examples3)]
            np.random.shuffle(index3)
            generatedX = generatedX[index3]
            '''
            for batch_idx, (inputs, targets) in enumerate(dataX):
                inputs = inputs.to(device)
                targets = targets.to(device)

                self.run_step_self(inputs)

            #self.Generate_Images(i)

    def train_self_bySingleData_Unsupervised(self,epoch,device,dataX):
        batchSize = self.batch_size
        smallBatchSize = int(batchSize/2.0)
        count = int(np.shape(dataX)[0] / smallBatchSize)

        for i in range(epoch):

            '''
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            n_examples3 = np.shape(generatedX)[0]
            index3 = [i for i in range(n_examples3)]
            np.random.shuffle(index3)
            generatedX = generatedX[index3]
            '''
            for batch_idx, inputs in enumerate(dataX):
                inputs = inputs.to(device)

                self.run_step_self(inputs)

            #self.Generate_Images(i)


    def train_self_byDataLoad(self,epoch,device,dataX,generatedX):
        batchSize = self.batch_size
        smallBatchSize = int(batchSize/2.0)

        for i in range(epoch):

            '''
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            n_examples3 = np.shape(generatedX)[0]
            index3 = [i for i in range(n_examples3)]
            np.random.shuffle(index3)
            generatedX = generatedX[index3]
            '''
            for batch_idx, (inputs, targets) in enumerate(dataX):

                inputs2 = 0
                targets2 = 0

                inputs2, targets2 = next(iter(generatedX))

                inputs = inputs.to(device)
                targets = targets.to(device)
                inputs2 = inputs2.to(device)
                targets2 = targets2.to(device)

                batchX = th.cat([inputs,inputs2],0)
                batchY = th.cat([targets,targets2],0)
                self.run_step_self(batchX)

            #self.Generate_Images(i)


    def train_Memory_Small(self,epoch,dataX):
        self.run_step_self(dataX)

    def train_Memory(self,epoch,dataX):
        batchSize = self.batch_size
        count = int(np.shape(dataX)[0] / batchSize)

        for cc in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j*batchSize:(j+1)*batchSize]
                self.run_step_self(batchX)

            #self.Generate_Images(i)

    def train_Memory_Numpy(self,epoch,dataX):
        batchSize = self.batch_size
        count = int(np.shape(dataX)[0] / batchSize)

        for cc in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j*batchSize:(j+1)*batchSize]
                #batchX = th.Tensor(batchX)
                #batchX.to(self.device)
                batchX = th.tensor(batchX).cuda().to(device=self.device, dtype=th.float)
                self.run_step_self(batchX)

            #self.Generate_Images(i)


    def LoadImageFromPath(self,path):
        batch = [GetImage_cv(
            sample_file,
            input_height=self.originalInputSize,
            input_width=self.originalInputSize,
            resize_height=self.inputImageSize,
            resize_width=self.inputImageSize,
            crop=False)
            for sample_file in path]
        return batch

    def train_Memory_Cpu_WithFiles(self,epoch,dataX,device,inputSize,originalSize):
        batchSize = self.batch_size
        self.originalInputSize = originalSize
        self.inputImageSize = inputSize
        count = int(np.shape(dataX)[0] / batchSize)

        for cc in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j*batchSize:(j+1)*batchSize]
                batchX = self.LoadImageFromPath(batchX)

                batchX = np.array(batchX)
                batchX = batchX.transpose(0, 3, 1, 2)

                batchX = th.tensor(batchX).cuda().to(device=device, dtype=th.float)

                self.run_step_self(batchX)

            #self.Generate_Images(i)

    def train_Memory_Cpu_WithFilesAndSize(self,epoch,dataX,device,inputSize):
        batchSize = self.batch_size
        count = int(np.shape(dataX)[0] / batchSize)
        self.inputImageSize = inputSize

        for cc in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j*batchSize:(j+1)*batchSize]
                batchX = self.LoadImageFromPath(batchX)

                batchX = np.array(batchX)
                batchX = batchX.transpose(0, 3, 1, 2)

                batchX = th.tensor(batchX).cuda().to(device=device, dtype=th.float)

                self.run_step_self(batchX)

            #self.Generate_Images(i)


    def train_Memory_Cpu(self,epoch,dataX,device):
        batchSize = self.batch_size
        count = int(np.shape(dataX)[0] / batchSize)

        for cc in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j*batchSize:(j+1)*batchSize]
                batchX = th.tensor(batchX).cuda().to(device=device, dtype=th.float)

                self.run_step_self(batchX)

            #self.Generate_Images(i)


    def train_TwoMemorys(self,epoch,dataX1,dataX2):
        batchSize = self.batch_size
        batchSize = int(batchSize/2)
        otherSize = self.batch_size - batchSize

        count = int(np.shape(dataX1)[0] / batchSize)
        count2 = int(np.shape(dataX2)[0] / otherSize)

        newMemory1 = dataX1
        newMemory2 = dataX2

        for cc in range(epoch):
            n_examples = np.shape(dataX1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            newMemory1 = newMemory1[index2]

            n_examples = np.shape(dataX2)[0]
            index3 = [i for i in range(n_examples)]
            np.random.shuffle(index3)
            newMemory2 = newMemory2[index3]

            for j in range(count):

                cj = j % count2
                batchX1 = newMemory1[j*batchSize:(j+1)*batchSize]
                batchX2 = newMemory2[j*otherSize:(j+1)*otherSize]
                myBatch = th.cat([batchX1,batchX2],0)

                self.run_step_self(myBatch)

            #self.Generate_Images(i)

    def LoadCACDFromPath(self,path):
        batch = [GetImage_cv(
            sample_file,
            input_height=250,
            input_width=250,
            resize_height=256,
            resize_width=256,
            crop=False)
            for sample_file in path]
        return batch


    def train_TwoMemorys_CACD(self,epoch,dataX1,dataX2,device):
        batchSize = self.batch_size
        batchSize = int(batchSize/2)
        otherSize = self.batch_size - batchSize

        count = int(np.shape(dataX1)[0] / batchSize)
        count2 = int(np.shape(dataX2)[0] / otherSize)

        newMemory1 = dataX1
        newMemory2 = dataX2

        for i in range(epoch):
            n_examples = np.shape(dataX1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            newMemory1 = newMemory1[index2]

            n_examples = np.shape(dataX2)[0]
            index3 = [i for i in range(n_examples)]
            np.random.shuffle(index3)
            newMemory2 = newMemory2[index3]

            for j in range(count):

                cj = j % count2
                batchX1 = newMemory1[j*batchSize:(j+1)*batchSize]
                batchX2 = newMemory2[cj*otherSize:(cj+1)*otherSize]

                batchImage = self.LoadCACDFromPath(batchX1)
                batchImage = th.tensor(batchImage).cuda().to(device=device, dtype=th.float)
                batchImage = batchImage.permute(0, 3, 1, 2)
                batchImage = batchImage.contiguous()

                batchImage2 = self.LoadCACDFromPath(batchX2)
                batchImage2 = th.tensor(batchImage2).cuda().to(device=device, dtype=th.float)
                batchImage2 = batchImage2.permute(0, 3, 1, 2)
                batchImage2 = batchImage2.contiguous()

                myBatch = th.cat([batchImage,batchImage2],0)

                self.run_step_self(myBatch)

            #self.Generate_Images(i)


    def train_MemoryAndGeneratedImages(self,epoch,dataX,teacher):
        batchSize = self.batch_size
        batchSize = int(batchSize / teacher.GetTeacherCount())
        otherSize = self.batch_size - batchSize*teacher.GetTeacherCount()

        count = int(np.shape(dataX)[0] / otherSize)
        tcount = int(np.shape(teacher.teacher.teacherArray[t1].currentKnowledge)[0] / batchSize)

        for cc in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j*otherSize:(j+1)*otherSize]

                c1 = j % tcount

                for t1 in range(teacher.GetTeacherCount()-1):
                    a1 = teacher.teacher.teacherArray[t1].currentKnowledge[c1*batchSize:(c1+1)*batchSize]
                    batchX = th.cat([batchX,a1],0)

                self.run_step_self(batchX)

            #self.Generate_Images(i)

    def train_Memory_System(self,epoch,dataX,device):
        batchSize = self.batch_size

        for cc in range(epoch):
            for step,batch_x in enumerate(dataX):
                batch_x = batch_x.to(device)
                self.run_step_self(batch_x)


    def train_self(self,epoch,dataX,generatedX):
        batchSize = self.batch_size
        smallBatchSize = int(batchSize/2.0)
        count = int(np.shape(dataX)[0] / smallBatchSize)
        myCount = int(np.shape(generatedX)[0] / smallBatchSize)

        for cc in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            n_examples3 = np.shape(generatedX)[0]
            index3 = [i for i in range(n_examples3)]
            np.random.shuffle(index3)
            generatedX = generatedX[index3]

            for j in range(count):
                batchX = dataX[j*smallBatchSize:(j+1)*smallBatchSize]
                myj = j % myCount
                generatedBatch = generatedX[myj*smallBatchSize:(myj+1)*smallBatchSize]
                newbatch = mytorch.cat([batchX,generatedBatch],dim=0)

                self.run_step_self(newbatch)

            #self.Generate_Images(i)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            self.Generate_Images(0)

            batch, cond = next(self.data)
            self.run_step(batch, cond)

            return
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            '''
            if self.step % 500 == 0:
                #show generations
                self.Generate_Images(self.step)
            '''
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def run_step_self(self, batch):
        self.forward_backward_self(batch)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward_self(self, batch):

        zero_grad(self.model_params)
        t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,
            t
        )

        #with self.ddp_model.no_sync():
            #losses = compute_losses()
        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()


    def forward_backward(self, batch, cond):

        #print(batch.shape[0])
        #print(self.microbatch)
        #print(np.shape(batch))

        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())


            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
