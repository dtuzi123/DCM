import numpy as np

#from NetworkModels.Teacher_Model_ import Teacher,Balance_Teacher
from NetworkModels.Teacher_Model_NoMPI_ import Balance_Teacher_NoMPI
from NetworkModels.VAE_Model_ import Balance_StudentModel
import torch.nn as nn
from NetworkModels.TFCL_Teacher_ import *
from improved_diffusion.train_util_balance_NoMPI_MultiGPU import *
from NetworkModels.VAE_Model_ import *

class TFCL_TeacherStudent(nn.Module):

    def __init__(self,name,device,input_size):
        super(TFCL_TeacherStudent, self).__init__()

        self.input_size = input_size
        self.teacher = TFCL_Teacher(input_size)
        self.student = TFCL_StudentModel(device,input_size)
        self.device = device
        self.trainingCount = 0
        self.trainingUpdate = 4
        self.GeneratingBatchSampleSize = 64
        self.batch_size = 64
        self.isTrainer = 0

    def CreateNewTeacher(self):
        newTeacher = self.teacher.Create_NewTeacher()
        self.teacher.currentTeacher = newTeacher

        for i in range(self.teacher.GetTeacherCount() - 1):
            currentTeacher = self.teacher.teacherArray[i]
            currentKnowledge = self.teacher.Sampling_By_Num(currentTeacher.diffusion, currentTeacher.model,
                                                            self.GeneratingBatchSampleSize)
            currentTeacher.currentKnowledge = currentKnowledge

        return newTeacher


    def GiveGeneration_ByTS(self,num):
        model = self.teacher.teacherArray[0].model
        diffusion = self.teacher.teacherArray[0].diffusion
        Truth,NoiseData = self.teacher.SamplingHalf_By_Num(diffusion, model, num)
        reco = self.student.Give_Reconstruction(NoiseData)
        return reco

    def GiveGeneration_ByTS_2(self,num):

        if num < self.batch_size:
            model = self.teacher.teacherArray[0].model
            diffusion = self.teacher.teacherArray[0].diffusion
            Truth,NoiseData = self.teacher.SamplingNoiseImages(diffusion,model,num,900)
            reco = self.student.Give_Reconstruction(NoiseData)
            return reco
        else:
            arr = []
            count = int(num / self.batch_size)
            for i in range(count):
                model = self.teacher.teacherArray[0].model
                diffusion = self.teacher.teacherArray[0].diffusion
                Truth, NoiseData = self.teacher.SamplingNoiseImages(diffusion, model, num, 900)
                reco = self.student.Give_Reconstruction(NoiseData)

                if np.shape(arr)[0] == 0:
                    arr = reco
                else:
                    torch.cat([arr,reco],0)
            return arr


    def GiveReconstruction_ByTS(self,batch):
        reco = 0
        arr = []
        if np.shape(batch)[0] < self.batch_size:
            noiseImage = self.teacher.q_sample(batch,500)
            reco = self.student.Give_Reconstruction(noiseImage)
            return reco
        else:
            count = int(np.shape(batch)[0]/self.batch_size)
            arr = []
            for i in range(count):
                smallBatch = batch[i*self.batch_size:(i+1)*self.batch_size]
                noiseImage = self.teacher.q_sample(smallBatch, 500)
                reco = self.student.Give_Reconstruction(noiseImage)

                if np.shape(arr)[0] == 0:
                    arr = reco
                else:
                    arr = torch.cat([arr,reco],0)

        return arr

    def GiveGeneration_ByTSAndStep(self,num,step):
        model = self.teacher.teacherArray[0].model
        diffusion = self.teacher.teacherArray[0].diffusion
        Truth,NoiseData = self.teacher.Sampling_By_NumAndStep(diffusion, model, num,step)
        NoiseData = NoiseData.to(self.device)
        reco = self.student.Give_Reconstruction(NoiseData)
        return reco


    def GiveReconstruction_ByTS(self,batch):
        model = self.teacher.teacherArray[0].model
        diffusion = self.teacher.teacherArray[0].diffusion
        Truth,NoiseData = self.teacher.SaplingHalf_By_Num_WithImages(diffusion, model, batch)
        reco = self.student.Give_Reconstruction(NoiseData)
        return reco

    def TrainStudent(self,epoch,memory):
        # using the KD
        if np.shape(self.teacher.teacherArray)[0] > 1:
            isKD = 1
        else:
            isKD = 0

        if isKD == 1:
            #Generating the knowledge
            if self.trainingCount % self.trainingUpdate == 0:
                for i in range(self.teacher.GetTeacherCount() - 1):
                    currentTeacher = self.teacher.teacherArray[i]
                    currentKnowledge = self.teacher.Sampling_By_Num(currentTeacher.diffusion,currentTeacher.model,self.GeneratingBatchSampleSize)
                    currentTeacher.currentKnowledge = currentKnowledge

            self.student.Train_Self_(epoch, memory,self.teacher)
        else:
            #Not using the KD
            self.student.Train_Self_Single_(epoch,memory)

        self.trainingCount = self.trainingCount + 1
        if self.trainingCount > 100000:
            self.trainingCount = 0

    def TrainStudent_FromTwoMemorys_WithBeta(self,epoch,memory1,memory2,beta):
        # using the KD
        if np.shape(self.teacher.teacherArray)[0] > 1:
            isKD = 1
        else:
            isKD = 0

        if isKD == 1:
            #Generating the knowledge
            if self.trainingCount % self.trainingUpdate == 0:
                for i in range(self.teacher.GetTeacherCount() - 1):
                    currentTeacher = self.teacher.teacherArray[i]
                    currentKnowledge = self.teacher.Sampling_By_Num(currentTeacher.diffusion,currentTeacher.model,self.GeneratingBatchSampleSize)
                    currentTeacher.currentKnowledge = currentKnowledge

            self.student.Train_Self_(epoch, memory,self.teacher)
        else:
            #Not using the KD
            self.student.Train_FromTwoMemorys_WithBeta(epoch,memory1,memory2,beta)

        self.trainingCount = self.trainingCount + 1
        if self.trainingCount > 100000:
            self.trainingCount = 0


    def TrainStudent_FromTwoMemorys(self,epoch,memory1,memory2):
        # using the KD
        if np.shape(self.teacher.teacherArray)[0] > 1:
            isKD = 1
        else:
            isKD = 0

        if isKD == 1:
            #Generating the knowledge
            if self.trainingCount % self.trainingUpdate == 0:
                for i in range(self.teacher.GetTeacherCount() - 1):
                    currentTeacher = self.teacher.teacherArray[i]
                    currentKnowledge = self.teacher.Sampling_By_Num(currentTeacher.diffusion,currentTeacher.model,self.GeneratingBatchSampleSize)
                    currentTeacher.currentKnowledge = currentKnowledge

            self.student.Train_Self_(epoch, memory,self.teacher)
        else:
            #Not using the KD
            self.student.Train_FromTwoMemorys(epoch,memory1,memory2)

        self.trainingCount = self.trainingCount + 1
        if self.trainingCount > 100000:
            self.trainingCount = 0

    def TrainTeacher(self,epoch,memory):
        args = self.teacher.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = []
        diffusion = self.teacher.currentTeacher.diffusion
        model = self.teacher.currentTeacher.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].model
        currentDiffusion = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].diffusion

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

    def TrainTeacher_FromTowMemorys(self,epoch,memory1,memory2):
        args = self.teacher.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = []
        diffusion = self.teacher.currentTeacher.diffusion
        model = self.teacher.currentTeacher.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].model
        currentDiffusion = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].diffusion

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

            self.trainer.train_TwoMemorys(epoch, memory1,memory2)
        else:
            self.trainer.train_TwoMemorys(epoch, memory1,memory2)


    def Train(self,epoch,memory):
        args = self.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        if self.isTrainer == 0:
            self.isTrainer = 1
            self.trainer = TrainLoop_Balance_NoMPI_MultiGPU(
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
            )  # .train_self(epoch, mydata,generatedData)

            self.trainer.train_Memory(epoch, memory, self.device)
        else:
            self.trainer.train_Memory(epoch, memory, self.device)



class TFCL_TeacherStudent256(nn.Module):

    def __init__(self,name,device,input_size):
        super(TFCL_TeacherStudent256, self).__init__()

        self.input_size = input_size
        self.teacher = TFCL_Teacher(input_size)
        self.student = TFCL_StudentModel256(device,input_size)
        self.device = device
        self.trainingCount = 0
        self.trainingUpdate = 4
        self.GeneratingBatchSampleSize = 64
        self.batch_size = 32
        self.isTrainer = 0

    def CreateNewTeacher(self):
        newTeacher = self.teacher.Create_NewTeacher()
        self.teacher.currentTeacher = newTeacher

        for i in range(self.teacher.GetTeacherCount() - 1):
            currentTeacher = self.teacher.teacherArray[i]
            currentKnowledge = self.teacher.Sampling_By_Num(currentTeacher.diffusion, currentTeacher.model,
                                                            self.GeneratingBatchSampleSize)
            currentTeacher.currentKnowledge = currentKnowledge

        return newTeacher


    def GiveGeneration_ByTS(self,num):
        model = self.teacher.teacherArray[0].model
        diffusion = self.teacher.teacherArray[0].diffusion
        Truth,NoiseData = self.teacher.SamplingHalf_By_Num(diffusion, model, num)
        reco = self.student.Give_Reconstruction(NoiseData)
        return reco


    def GiveGeneration_ByTSAndStep(self,num,step):
        model = self.teacher.teacherArray[0].model
        diffusion = self.teacher.teacherArray[0].diffusion
        Truth,NoiseData = self.teacher.Sampling_By_NumAndStep(diffusion, model, num,step)
        NoiseData = NoiseData.to(self.device)
        reco = self.student.Give_Reconstruction(NoiseData)
        return reco


    def GiveReconstruction_ByTS(self,batch):
        model = self.teacher.teacherArray[0].model
        diffusion = self.teacher.teacherArray[0].diffusion
        Truth,NoiseData = self.teacher.SaplingHalf_By_Num_WithImages(diffusion, model, batch)
        reco = self.student.Give_Reconstruction(NoiseData)
        return reco

    def TrainStudent(self,epoch,memory):
        # using the KD
        if np.shape(self.teacher.teacherArray)[0] > 1:
            isKD = 1
        else:
            isKD = 0

            self.student.Train_Self_Single_(epoch,memory)

        self.trainingCount = self.trainingCount + 1
        if self.trainingCount > 100000:
            self.trainingCount = 0


    def TrainStudent_FromTwoMemorys_WithBeta(self,epoch,memory1,memory2,beta):
        # using the KD
        if np.shape(self.teacher.teacherArray)[0] > 1:
            isKD = 1
        else:
            isKD = 0

        if isKD == 1:
            #Generating the knowledge
            if self.trainingCount % self.trainingUpdate == 0:
                for i in range(self.teacher.GetTeacherCount() - 1):
                    currentTeacher = self.teacher.teacherArray[i]
                    currentKnowledge = self.teacher.Sampling_By_Num(currentTeacher.diffusion,currentTeacher.model,self.GeneratingBatchSampleSize)
                    currentTeacher.currentKnowledge = currentKnowledge

            self.student.Train_Self_(epoch, memory,self.teacher)
        else:
            #Not using the KD
            self.student.Train_FromTwoMemorys_WithBeta(epoch,memory1,memory2,beta)

        self.trainingCount = self.trainingCount + 1
        if self.trainingCount > 100000:
            self.trainingCount = 0


    def TrainStudent_FromTwoMemorys_CACD(self,epoch,memory1,memory2):
        # using the KD

        self.student.Train_FromTwoMemorys_CACD(epoch,memory1,memory2)

        self.trainingCount = self.trainingCount + 1
        if self.trainingCount > 100000:
            self.trainingCount = 0


    def TrainStudent_FromTwoMemorys(self,epoch,memory1,memory2):
        # using the KD
        if np.shape(self.teacher.teacherArray)[0] > 1:
            isKD = 1
        else:
            isKD = 0

        if isKD == 1:
            #Generating the knowledge
            if self.trainingCount % self.trainingUpdate == 0:
                for i in range(self.teacher.GetTeacherCount() - 1):
                    currentTeacher = self.teacher.teacherArray[i]
                    currentKnowledge = self.teacher.Sampling_By_Num(currentTeacher.diffusion,currentTeacher.model,self.GeneratingBatchSampleSize)
                    currentTeacher.currentKnowledge = currentKnowledge

            self.student.Train_Self_(epoch, memory,self.teacher)
        else:
            #Not using the KD
            self.student.Train_FromTwoMemorys(epoch,memory1,memory2)

        self.trainingCount = self.trainingCount + 1
        if self.trainingCount > 100000:
            self.trainingCount = 0


    def TrainTeacher(self,epoch,memory):
        args = self.teacher.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = []
        diffusion = self.teacher.currentTeacher.diffusion
        model = self.teacher.currentTeacher.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].model
        currentDiffusion = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].diffusion

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

    def TrainTeacher_FromTowMemorys_CACD(self,epoch,memory1,memory2):
        args = self.teacher.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = []
        diffusion = self.teacher.currentTeacher.diffusion
        model = self.teacher.currentTeacher.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].model
        currentDiffusion = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].diffusion

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

            self.trainer.train_TwoMemorys_CACD(epoch, memory1,memory2,self.device)
        else:
            self.trainer.train_TwoMemorys_CACD(epoch, memory1,memory2,self.device)


    def TrainTeacher_FromTowMemorys(self,epoch,memory1,memory2):
        args = self.teacher.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = []
        diffusion = self.teacher.currentTeacher.diffusion
        model = self.teacher.currentTeacher.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        currentTeacher = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].model
        currentDiffusion = self.teacher.teacherArray[self.teacher.GetTeacherCount()-1].diffusion

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

            self.trainer.train_TwoMemorys(epoch, memory1,memory2)
        else:
            self.trainer.train_TwoMemorys(epoch, memory1,memory2)


    def Train(self,epoch,memory):
        args = self.args
        logger.log("creating data loader...")

        print("test")
        print(self.input_size)

        data = []
        diffusion = self.diffusion
        model = self.model
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        args.batch_size = self.batch_size
        args.microbatch = args.batch_size

        if self.isTrainer == 0:
            self.isTrainer = 1
            self.trainer = TrainLoop_Balance_NoMPI_MultiGPU(
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
            )  # .train_self(epoch, mydata,generatedData)

            self.trainer.train_Memory(epoch, memory, self.device)
        else:
            self.trainer.train_Memory(epoch, memory, self.device)

