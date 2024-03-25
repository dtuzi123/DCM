from models import VanillaVAE_,Autoencoder_
from torch import optim
import numpy as np
import torch
import cv2
import os
from skimage import io, transform
from cv2_imageProcess import *
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn as nn
from models.VAE256 import *
from models.wae_mmd import *
from models.vanilla_vae import *

'''
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(inTrain_One2_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)
'''

class Autoencoder(nn.Module):
    def __init__(self,device,inputSize):
        super(Autoencoder, self).__init__()

        print("build the student model")

        self.device = device
        self.input_size = inputSize

        if inputSize >= 128:
            self.vae = VAE(image_size=inputSize)
        else:
            self.vae = Autoencoder_(in_channels = 3,latent_dim=128,input_size=inputSize)

        #self.vae = VAE()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)
        self.batch_size = 64
        self.OriginalInputSize = 256

    def Train_Self_Single_Beta3_Numpy(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]
                batchX = torch.tensor(batchX).cuda().to(device=self.device, dtype=torch.float)

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()


    def training_step(self, batch):

        results = self.vae.forward(batch)

        train_loss = self.vae.loss_function(*results)

        return train_loss['loss']

    def Train_Cpu_WithFiles(self,epoch, memoryBuffer):
        batchsize = 128
        model = self.vae
        dataX = memoryBuffer
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                batchImage = self.LoadCACDFromPath(batchX)
                batchImage = torch.tensor(batchImage).cuda().to(device=device, dtype=torch.float)
                batchImage = batchImage.permute(0, 3, 1, 2)
                batchImage = batchImage.contiguous()
                batchX = batchImage

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

    def Train(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

    def LoadCACDFromPath(self,path):
        batch = [GetImage_cv(
            sample_file,
            input_height=self.OriginalInputSize,
            input_width=self.OriginalInputSize,
            resize_height=self.input_size,
            resize_width=self.input_size,
            crop=False)
            for sample_file in path]
        return batch

    def Train_Files(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                batchImage = self.LoadCACDFromPath(batchX)
                batchImage = torch.tensor(batchImage).cuda().to(device=device, dtype=torch.float)
                batchImage = batchImage.permute(0, 3, 1, 2)
                batchImage = batchImage.contiguous()
                batchX = batchImage

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

    def Give_Reconstruction(self,x):
        arr = []
        batchSize = 64
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i*batchSize:(i+1)*batchSize]
            myReco = self.vae.generate(batch)
            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr,myReco],dim=0)

        return arr

    def Give_ReconstructionSingle(self,x):
        myReco = self.vae.generate(x)
        return myReco

    def Give_LatentCode(self,x):
        return self.vae.GiveCode(x)

    def Give_Generations(self,n):
        return self.vae.generate(n)

    def Give_GenerationsWithN(self,n):
        return self.vae.generate_withN(n,self.device)

class Autoencoder196(nn.Module):
    def __init__(self,device,inputSize):
        super(Autoencoder196, self).__init__()

        print("build the student model")

        self.device = device

        self.vae = Autoencoder196_(in_channels = 3,latent_dim=128,input_size=inputSize)

        #self.vae = VAE()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)
        self.batch_size = 64

    def training_step(self, batch):

        results = self.vae.forward(batch)

        train_loss = self.vae.loss_function(*results)

        return train_loss['loss']

    def Train(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

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

    def Give_FullSampleByLatent(self,latentSamples):
        batch = latentSamples
        batch = batch.view(np.shape(batch)[0], -1)
        x = self.vae.decode(batch)
        return x

    def Give_SampleByLatent(self,latentSamples):
        batchSize = 64
        count = int(np.shape(latentSamples)[0] / batchSize)
        arr = []
        for i in range(count):
            batch = latentSamples[i*batchSize:(i+1)*batchSize]
            batch = batch.view(np.shape(batch)[0],-1)
            x = self.vae.decode(batch)
            if np.shape(arr)[0] == 0:
                arr = x
            else:
                arr = torch.cat([arr,x],0)

        return arr


    def Train_Files(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                batchImage = self.LoadCACDFromPath(batchX)
                batchImage = torch.tensor(batchImage).cuda().to(device=device, dtype=torch.float)
                batchImage = batchImage.permute(0, 3, 1, 2)
                batchImage = batchImage.contiguous()
                batchX = batchImage

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

    def Give_Reconstruction(self,x):
        arr = []
        batchSize = 64
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i*batchSize:(i+1)*batchSize]
            myReco = self.vae.generate(batch)
            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr,myReco],dim=0)

        return arr

    def Give_ReconstructionSingle(self,x):
        myReco = self.vae.generate(x)
        return myReco

    def Give_LatentCode(self,x):
        return self.vae.GiveCode(x)

    def Give_Generations(self,n):
        self.vae.generate()


class VAE196(nn.Module):
    def __init__(self,device,inputSize):
        super(VAE196, self).__init__()

        print("build the student model")

        self.device = device

        self.vae = VAE196_(in_channels = 3,latent_dim=128,input_size=inputSize)

        #self.vae = VAE()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)
        self.batch_size = 64

    def training_step(self, batch):

        results = self.vae.forward(batch)

        train_loss = self.vae.loss_function(*results)

        return train_loss['loss']

    def Train(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()


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

    def Give_FullSampleByLatent(self,latentSamples):
        batch = latentSamples
        batch = batch.view(np.shape(batch)[0], -1)
        x = self.vae.decode(batch)
        return x

    def Give_SampleByLatent(self,latentSamples):
        batchSize = 64
        count = int(np.shape(latentSamples)[0] / batchSize)
        arr = []
        for i in range(count):
            batch = latentSamples[i*batchSize:(i+1)*batchSize]
            batch = batch.view(np.shape(batch)[0],-1)
            x = self.vae.decode(batch)
            if np.shape(arr)[0] == 0:
                arr = x
            else:
                arr = torch.cat([arr,x],0)

        return arr


    def Train_Files(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                batchImage = self.LoadCACDFromPath(batchX)
                batchImage = torch.tensor(batchImage).cuda().to(device=device, dtype=torch.float)
                batchImage = batchImage.permute(0, 3, 1, 2)
                batchImage = batchImage.contiguous()
                batchX = batchImage

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

    def Give_Reconstruction(self,x):
        arr = []
        batchSize = 64
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i*batchSize:(i+1)*batchSize]
            myReco = self.vae.generate(batch)
            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr,myReco],dim=0)

        return arr

    def Give_ReconstructionSingle(self,x):
        myReco = self.vae.generate(x)
        return myReco

    def Give_LatentCode(self,x):
        return self.vae.GiveCode(x)

    def Give_Generations(self,n):
        self.vae.generate()


class StudentModel(nn.Module):
    def __init__(self,device,inputSize):
        super(StudentModel, self).__init__()

        print("build the student model")

        self.device = device

        self.vae = VanillaVAE_(in_channels = 3,latent_dim=128,input_size=inputSize)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)
        self.batch_size = 64
        self.originalInputSize = 256


    def ComputerLoss(self,data):
        results = self.forward(data, labels=None)
        train_loss = self.vae.loss_function(*results)
        return train_loss

    def Give_MeanAndVar(self,tensor):
        return self.vae.Give_MeanAndVar(tensor)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.vae(input, **kwargs)

    def Train_Self_Single_Beta3_Numpy(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]
                batchX = torch.tensor(batchX).cuda().to(device=self.device, dtype=torch.float)

                self.optimizer.zero_grad()
                loss = self.training_step_withBeta3(batchX)
                loss.backward()
                self.optimizer.step()


    def training_step(self, batch):

        results = self.forward(batch, labels=None)

        train_loss = self.vae.loss_function(*results)

        return train_loss['loss']

    def training_step_withBeta3(self, batch):

        results = self.forward(batch, labels=None)
        results.append(batch)
        results.append(0.5)

        train_loss = self.vae.loss_function_WithBeta(*results)

        return train_loss['loss']

    def training_step2(self, batch,batch2,beta):

        results = self.forward(batch, labels=None)
        results.append(batch2)
        results.append(beta)

        train_loss = self.vae.loss_function_WithBeta(*results)

        return train_loss['loss']


    def training_step2_WithBeta(self, batch,batch2,beta):

        results = self.forward(batch, labels=None)
        results.append(batch2)
        results.append(beta)

        train_loss = self.vae.loss_function_WithBeta(*results)

        return train_loss['loss']


    def GiveGeneration(self,n):
        batch = 64
        count = int(n/batch)
        arr = []
        for i in range(count):
            b = self.vae.sample(batch, self.device)
            if np.shape(arr)[0] == 0:
                arr = b
            else:
                arr = torch.cat([arr,b],0)
        return arr

    def Generation(self,n):
        return self.vae.sample(n,self.device)


    def Give_ReconstructionSingle(self,x):
        myReco = self.vae.generate(x)
        return myReco

    def Give_Reconstruction(self,x):
        arr = []
        batchSize = 64
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i*batchSize:(i+1)*batchSize]
            myReco = self.vae.generate(batch)
            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr,myReco],dim=0)

        return arr

    def Train_One2(self,sample,sample2,beta):
        self.optimizer.zero_grad()
        loss = self.training_step2(sample,sample2,beta)
        loss.backward()
        self.optimizer.step()

    def Train_One(self,sample):
        self.optimizer.zero_grad()
        loss = self.training_step(sample)
        loss.backward()
        self.optimizer.step()

    def Train_Self(self,epoch,data):

        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        count = int(np.shape(data)[0] / batchsize)
        for i in range(epoch):
            for j in range(count):
                if j > 0:
                    optimizer.zero_grad()
                batchX = data[j*batchsize:(j+1)*batchsize]
                loss = self.training_step(batchX)
                loss.backward()
                optimizer.step()

            '''
            print("Generation")
            #a1 = self.Generation(10)
            a1 = self.Give_Reconstruction(data[0:10])

            
            a1 = a1.unsqueeze(0).cuda().cpu()
            a1 = a1.detach().numpy()
            a1 = a1[0]
            a1 = np.transpose(a1, (0, 2, 3, 1))
            a1 = (a1 + 1) * 127.5

            out1 = merge2(a1[:10], [1, 10])

            name = "bbbb" + str(i) + ".png"
            cv2.imwrite("/scratch/fy689/improved-diffusion-main/results/" + name, out1)
            cv2.waitKey(0)
            '''

class StudentModel_VAEMMD(nn.Module):
    def __init__(self, device, inputSize):
        super(StudentModel_VAEMMD, self).__init__()

        print("build the student model")

        self.device = device
        self.vae = WAE_MMD(in_channels=3, latent_dim=128, input_size=inputSize)#VanillaVAE_(in_channels=3, latent_dim=128, input_size=inputSize)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)
        self.batch_size = 64

    def ComputerLoss(self, data):
        results = self.forward(data, labels=None)
        train_loss = self.vae.loss_function(*results)
        return train_loss

    def Give_MeanAndVar(self, tensor):
        return self.vae.Give_MeanAndVar(tensor)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.vae(input, **kwargs)

    def training_step(self, batch):

        results = self.forward(batch, labels=None)

        train_loss = self.vae.loss_function(*results)

        return train_loss['loss']

    def training_step2(self, batch, batch2, beta):

        results = self.forward(batch, labels=None)
        results.append(batch2)
        results.append(beta)

        train_loss = self.vae.loss_function_WithBeta(*results)

        return train_loss['loss']

    def training_step2_WithBeta(self, batch, batch2, beta):

        results = self.forward(batch, labels=None)
        results.append(batch2)
        results.append(beta)

        train_loss = self.vae.loss_function_WithBeta(*results)

        return train_loss['loss']

    def Generation(self, n):
        return self.vae.sample(n, self.device)

    def Give_ReconstructionSingle(self, x):
        myReco = self.vae.generate(x)
        return myReco

    def Give_Reconstruction(self, x):
        arr = []
        batchSize = 64
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i * batchSize:(i + 1) * batchSize]
            myReco = self.vae.generate(batch)
            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr, myReco], dim=0)

        return arr

    def Train_One2(self, sample, sample2, beta):
        self.optimizer.zero_grad()
        loss = self.training_step2(sample, sample2, beta)
        loss.backward()
        self.optimizer.step()

    def Train_One(self, sample):
        self.optimizer.zero_grad()

        loss = self.training_step(sample)
        loss.backward()
        self.optimizer.step()

    def Train_Self(self, epoch, data):
        #print(np.shape(data))

        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        count = int(np.shape(data)[0] / batchsize)
        for i in range(epoch):
            for j in range(count):
                if j > 0:
                    optimizer.zero_grad()
                batchX = data[j * batchsize:(j + 1) * batchsize]
                loss = self.training_step(batchX)
                loss.backward()
                optimizer.step()
                #print(loss)

            '''
            print("Generation")
            #a1 = self.Generation(10)
            a1 = self.Give_Reconstruction(data[0:10])

            print(np.shape(a1))

            a1 = a1.unsqueeze(0).cuda().cpu()
            a1 = a1.detach().numpy()
            a1 = a1[0]
            print(np.shape(a1))
            a1 = np.transpose(a1, (0, 2, 3, 1))
            a1 = (a1 + 1) * 127.5

            out1 = merge2(a1[:10], [1, 10])

            name = "bbbb" + str(i) + ".png"
            cv2.imwrite("/scratch/fy689/improved-diffusion-main/results/" + name, out1)
            cv2.waitKey(0)
            '''

class Balance_StudentModel(StudentModel):

    def Train_Self_ByDataLoad_Single(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        for i in range(epoch):
            for batch_idx, (inputs, targets) in enumerate(dataX):

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                batchX = inputs
                batchY = targets

                optimizer.zero_grad()

                loss = self.training_step(batchX)
                loss.backward()
                optimizer.step()
                #print(loss)


    def Train_Self_ByDataLoad(self, epoch, data,generatedData):
        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        for i in range(epoch):
            for batch_idx, (inputs, targets) in enumerate(data):
                batch_idx2, (inputs2, targets2) = enumerate(generatedData)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                inputs2 = inputs2.to(self.device)
                targets2 = targets2.to(self.device)

                batchX = th.cat([inputs,inputs2],0)
                batchY = th.cat([targets,targets2],0)

                optimizer.zero_grad()

                loss = self.training_step(batchX)
                loss.backward()
                optimizer.step()
                #print(loss)

    def Train_Self_(self, epoch, data,generatedData):
        #print(np.shape(data))

        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        smallBatch = int(batchsize/2.0)
        count = int(np.shape(data)[0] / smallBatch)
        myCount = int(np.shape(generatedData)[0] / smallBatch)

        for i in range(epoch):
            for j in range(count):
                if j > 0:
                    optimizer.zero_grad()
                batchX = data[j * smallBatch:(j + 1) * smallBatch]
                myj = j % myCount
                generatedX = generatedData[myj * smallBatch:(myj + 1) * smallBatch]

                newX = torch.cat([batchX,generatedX],dim = 0)

                loss = self.training_step(newX)
                loss.backward()
                optimizer.step()
                #print(loss)

    def Train_Self_WithBeta_Single(self, epoch, data,beta):
        #print(np.shape(data))

        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        myCount = int(np.shape(data)[0] / self.batch_size)

        for i in range(epoch):
            for j in range(myCount):
                if j > 0:
                    optimizer.zero_grad()
                batchX = data[j * self.batch_size:(j + 1) * self.batch_size]

                newX = batchX

                loss = self.training_step2_WithBeta(newX,newX,beta)
                loss.backward()
                optimizer.step()
                #print(loss)


    def Train_Self_WithBeta_Single_Cpu(self, epoch, data,beta):
        #print(np.shape(data))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        myCount = int(np.shape(data)[0] / self.batch_size)

        for i in range(epoch):
            for j in range(myCount):
                if j > 0:
                    optimizer.zero_grad()
                batchX = data[j * self.batch_size:(j + 1) * self.batch_size]

                newX = batchX
                newX = torch.tensor(newX).cuda().to(device=device, dtype=torch.float)

                loss = self.training_step2_WithBeta(newX,newX,beta)
                loss.backward()
                optimizer.step()
                #print(loss)


    def Train_Self_WithBeta(self, epoch, data,generatedData,beta):
        #print(np.shape(data))

        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        smallBatch = int(batchsize/2.0)
        count = int(np.shape(data)[0] / smallBatch)
        myCount = int(np.shape(generatedData)[0] / smallBatch)

        for i in range(epoch):
            for j in range(count):
                if j > 0:
                    optimizer.zero_grad()
                batchX = data[j * smallBatch:(j + 1) * smallBatch]
                myj = j % myCount
                generatedX = generatedData[myj * smallBatch:(myj + 1) * smallBatch]

                newX = torch.cat([batchX,generatedX],dim = 0)

                loss = self.training_step2_WithBeta(newX,newX,beta)
                loss.backward()
                optimizer.step()
                #print(loss)


    def Train_Self_WithBeta_Cpu(self, epoch, data,generatedData,beta):
        #print(np.shape(data))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        smallBatch = int(batchsize/2.0)
        count = int(np.shape(data)[0] / smallBatch)
        myCount = int(np.shape(generatedData)[0] / smallBatch)

        for i in range(epoch):
            for j in range(count):
                if j > 0:
                    optimizer.zero_grad()
                batchX = data[j * smallBatch:(j + 1) * smallBatch]
                myj = j % myCount
                generatedX = generatedData[myj * smallBatch:(myj + 1) * smallBatch]

                #newX = torch.cat([batchX,generatedX],dim = 0)
                newX = np.concatenate((batchX,generatedX),0)
                newX = torch.tensor(newX).cuda().to(device=device, dtype=torch.float)

                loss = self.training_step2_WithBeta(newX,newX,beta)
                loss.backward()
                optimizer.step()
                #print(loss)


    def Give_Reconstruction_NoGPU(self,x):
        arr = []
        batchSize = self.batch_size
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i*batchSize:(i+1)*batchSize]
            batch = torch.tensor(batch).cuda().to(device=self.device, dtype=torch.float)
            myReco = self.vae.generate(batch)
            myReco = myReco.unsqueeze(0).cuda().cpu()

            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr,myReco],dim=0)

        return arr

    def Give_LatentCode(self,batch):
        mean,var = self.vae.Give_MeanAndVar(batch)
        z = self.vae.reparameterize(mean,var)
        return z

    def GenerateFromLatentCode(self,z):
        x = self.vae.decode(z)
        return x

class TFCL_StudentModel(StudentModel):

    def Give_LatentCode(self,batch):
        mean,var = self.vae.Give_MeanAndVar(batch)
        z = self.vae.reparameterize(mean,var)
        return z

    def GenerateFromLatentCode(self,z):
        x = self.vae.decode(z)
        return x

    def Train_Self_ByDataLoad_Single(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        for i in range(epoch):
            for batch_idx, inputs in enumerate(dataX):

                inputs = inputs.to(self.device)

                batchX = inputs

                optimizer.zero_grad()

                loss = self.training_step(batchX)
                loss.backward()
                optimizer.step()
                #print(loss)


    def TrainStep(self,batch):
        self.optimizer.zero_grad()
        loss = self.training_step(batch)
        loss.backward()
        self.optimizer.step()

    def Train_Self_Single_WithBeta(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()


    def Train_Self_Single_(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()


    def Train_Self_Single_Beta3_Cpu_WithBeta(self, epoch, dataX,beta):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]
                batchX = torch.tensor(batchX).cuda().to(device=device, dtype=torch.float)

                self.optimizer.zero_grad()
                loss = self.training_step2_WithBeta(batchX,batchX,beta)
                loss.backward()
                self.optimizer.step()


    def Train_Self_Single_Beta3_Cpu(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]
                batchX = torch.tensor(batchX).cuda().to(device=device, dtype=torch.float)

                self.optimizer.zero_grad()
                loss = self.training_step_withBeta3(batchX)
                loss.backward()
                self.optimizer.step()


    def Train_Self_Single_Beta3(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                self.optimizer.zero_grad()
                loss = self.training_step_withBeta3(batchX)
                loss.backward()
                self.optimizer.step()


    def Train_Self_Single_Beta3_Numpy(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]
                batchX = torch.tensor(batchX).cuda().to(device=self.device, dtype=torch.float)

                self.optimizer.zero_grad()
                loss = self.training_step_withBeta3(batchX)
                loss.backward()
                self.optimizer.step()


    def Train_Self_Single_Beta3_Balance_Numpy(self, epoch,dataX1 ,dataX):
        batchsize = 128
        model = self.vae
        otherSize = 64
        count = int(np.shape(dataX)[0] / otherSize)
        count2 = int(np.shape(dataX1)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            n_examples = np.shape(dataX1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX1 = dataX1[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]
                cj = j % count2

                batchX1 = dataX1[cj * otherSize:(cj + 1) * otherSize]
                batchX = np.concatenate((batchX,batchX1),0)

                batchX = torch.tensor(batchX).cuda().to(device=self.device, dtype=th.float)

                self.optimizer.zero_grad()
                loss = self.training_step_withBeta3(batchX)
                loss.backward()
                self.optimizer.step()


    def Give_Reconstruction(self,x):
        arr = []
        batchSize = 64
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i*batchSize:(i+1)*batchSize]
            myReco = self.vae.generate(batch)
            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr,myReco],dim=0)

        return arr

    def Give_Reconstruction_Single(self,x):
        myReco = self.vae.generate(x)
        return myReco

    def Generation(self,n):
        return self.vae.sample(n,self.device)

    def Train_FromTwoMemorys_WithBeta(self, epoch, memory1,memory2,beta):
        batchsize = 128
        model = self.vae

        batchSize = self.batch_size
        batchSize = int(batchSize / 2)
        otherSize = self.batch_size - batchSize

        count = int(np.shape(memory1)[0] / batchSize)
        count2 = int(np.shape(memory2)[0] / otherSize)

        newMemory1 = memory1
        newMemory2 = memory2

        for i in range(epoch):
            n_examples = np.shape(newMemory1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            newMemory1 = newMemory1[index2]

            n_examples = np.shape(newMemory2)[0]
            index3 = [i for i in range(n_examples)]
            np.random.shuffle(index3)
            newMemory2 = newMemory2[index3]

            for j in range(count):
                batchX = newMemory1[j * batchSize:(j + 1) * batchSize]

                c1 = j % count2
                batchX2 = newMemory2[c1 * otherSize:(c1 + 1) * otherSize]

                batchX = torch.cat([batchX,batchX2],0)

                self.optimizer.zero_grad()
                loss = self.training_step2_WithBeta(batchX,batchX,beta)
                loss.backward()
                self.optimizer.step()

    def Train_FromTwoMemorys(self, epoch, memory1,memory2):
        batchsize = 128
        model = self.vae

        batchSize = self.batch_size
        batchSize = int(batchSize / 2)
        otherSize = self.batch_size - batchSize

        count = int(np.shape(memory1)[0] / batchSize)
        count2 = int(np.shape(memory2)[0] / otherSize)

        newMemory1 = memory1
        newMemory2 = memory2

        for i in range(epoch):
            n_examples = np.shape(newMemory1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            newMemory1 = newMemory1[index2]

            n_examples = np.shape(newMemory2)[0]
            index3 = [i for i in range(n_examples)]
            np.random.shuffle(index3)
            newMemory2 = newMemory2[index3]

            for j in range(count):
                batchX = newMemory1[j * batchSize:(j + 1) * batchSize]

                c1 = j % count2
                batchX2 = newMemory2[c1 * otherSize:(c1 + 1) * otherSize]

                batchX = torch.cat([batchX,batchX2],0)

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

    def Train_Self_(self, epoch, dataX,teacher):
        batchsize = 128
        model = self.vae

        batchSize = self.batch_size
        batchSize = int(batchSize / teacher.GetTeacherCount())
        otherSize = self.batch_size - batchSize * (teacher.GetTeacherCount()-1)

        count = int(np.shape(dataX)[0] / otherSize)
        tcount = int(np.shape(teacher.teacherArray[0].currentKnowledge)[0] / batchSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                c1 = j % tcount

                for t1 in range(teacher.GetTeacherCount() - 1):
                    a1 = teacher.teacherArray[t1].currentKnowledge[c1 * batchSize:(c1 + 1) * batchSize]
                    batchX = torch.cat([batchX, a1], 0)

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

class TFCL_StudentModel256(nn.Module):
    def __init__(self,device,inputSize,originalInputSize):
        super(TFCL_StudentModel256, self).__init__()

        print("build the student model")

        self.input_size = inputSize
        self.device = device

        self.OriginalInputSize = originalInputSize


        self.vae = VAE(image_size=inputSize)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(device)

        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)
        self.batch_size = 32

    def LoadCACDFromPath(self,path):
        batch = [GetImage_cv(
            sample_file,
            input_height=self.OriginalInputSize,
            input_width=self.OriginalInputSize,
            resize_height=self.input_size,
            resize_width=self.input_size,
            crop=False)
            for sample_file in path]
        return batch

    def Train_Self_ByDataLoad_Single(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.0001)

        for i in range(epoch):
            for batch_idx, inputs in enumerate(dataX):

                inputs = inputs.to(self.device)

                batchX = inputs

                optimizer.zero_grad()

                loss = self.training_step(batchX)
                loss.backward()
                optimizer.step()
                #print(loss)

    def Train_Self_Single_(self, epoch, dataX):
        batchsize = 128
        model = self.vae
        otherSize = self.batch_size
        count = int(np.shape(dataX)[0] / otherSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]
                self.vae.Update(batchX)
                '''
                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()
                '''

    def Give_Reconstruction(self,x):
        arr = []
        batchSize = self.batch_size
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i*batchSize:(i+1)*batchSize]
            myReco = self.vae.Give_Reco(batch)
            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr,myReco],dim=0)

        return arr

    def Give_Reconstruction_NoGPU(self,x):
        arr = []
        batchSize = self.batch_size
        count = int(np.shape(x)[0] / batchSize)
        for i in range(count):
            batch = x[i*batchSize:(i+1)*batchSize]
            batch = torch.tensor(batch).cuda().to(device=self.device, dtype=torch.float)
            myReco = self.vae.Give_Reco(batch)
            myReco = myReco.unsqueeze(0).cuda().cpu()

            if np.shape(arr)[0] == 0:
                arr = myReco
            else:
                arr = torch.cat([arr,myReco],dim=0)

        return arr

    def Generation(self,n):
        return self.vae.sample(n,self.device)

    def Train_FromTwoMemorys_WithBeta(self, epoch, memory1,memory2,beta):
        batchsize = 128
        model = self.vae

        batchSize = self.batch_size
        batchSize = int(batchSize / 2)
        otherSize = self.batch_size - batchSize

        count = int(np.shape(memory1)[0] / batchSize)
        count2 = int(np.shape(memory2)[0] / otherSize)

        newMemory1 = memory1
        newMemory2 = memory2

        for i in range(epoch):
            n_examples = np.shape(newMemory1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            newMemory1 = newMemory1[index2]

            n_examples = np.shape(newMemory2)[0]
            index3 = [i for i in range(n_examples)]
            np.random.shuffle(index3)
            newMemory2 = newMemory2[index3]

            for j in range(count):
                batchX = newMemory1[j * batchSize:(j + 1) * batchSize]

                c1 = j % count2
                batchX2 = newMemory2[c1 * otherSize:(c1 + 1) * otherSize]

                batchX = torch.cat([batchX,batchX2],0)

                self.optimizer.zero_grad()
                loss = self.training_step2_WithBeta(batchX,batchX,beta)
                loss.backward()
                self.optimizer.step()

    def Train_Files(self,epoch,memory):
        batchsize = 128
        model = self.vae

        batchSize = self.batch_size

        count = int(np.shape(memory)[0] / batchSize)

        newMemory1 = memory

        for i in range(epoch):
            n_examples = np.shape(newMemory1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            newMemory1 = newMemory1[index2]

            for j in range(count):
                batchX = newMemory1[j * batchSize:(j + 1) * batchSize]

                batchImage = self.LoadCACDFromPath(batchX)
                batchImage = torch.tensor(batchImage).cuda().to(device=self.device, dtype=torch.float)
                batchImage = batchImage.permute(0, 3, 1, 2)
                batchImage = batchImage.contiguous()
                batchX = batchImage

                self.vae.Update(batchX)


    def Train_FromTwoMemorys_CACD(self, epoch, memory1,memory2):
        batchsize = 128
        model = self.vae

        batchSize = self.batch_size
        batchSize = int(batchSize / 2)
        otherSize = self.batch_size - batchSize

        count = int(np.shape(memory1)[0] / batchSize)
        count2 = int(np.shape(memory2)[0] / otherSize)

        newMemory1 = memory1
        newMemory2 = memory2

        for i in range(epoch):
            n_examples = np.shape(newMemory1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            newMemory1 = newMemory1[index2]

            n_examples = np.shape(newMemory2)[0]
            index3 = [i for i in range(n_examples)]
            np.random.shuffle(index3)
            newMemory2 = newMemory2[index3]

            for j in range(count):
                batchX = newMemory1[j * batchSize:(j + 1) * batchSize]

                batchImage = self.LoadCACDFromPath(batchX)
                batchImage = torch.tensor(batchImage).cuda().to(device=self.device, dtype=torch.float)
                batchImage = batchImage.permute(0, 3, 1, 2)
                batchImage = batchImage.contiguous()

                c1 = j % count2
                batchX2 = newMemory2[c1 * otherSize:(c1 + 1) * otherSize]

                batchImage2 = self.LoadCACDFromPath(batchX2)
                batchImage2 = torch.tensor(batchImage2).cuda().to(device=self.device, dtype=torch.float)
                batchImage2 = batchImage2.permute(0, 3, 1, 2)
                batchImage2 = batchImage2.contiguous()

                batchX = torch.cat([batchImage,batchImage2],0)
                self.vae.Update(batchX)
                '''
                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()
                '''


    def Train_FromTwoMemorys(self, epoch, memory1,memory2):
        batchsize = 128
        model = self.vae

        batchSize = self.batch_size
        batchSize = int(batchSize / 2)
        otherSize = self.batch_size - batchSize

        count = int(np.shape(memory1)[0] / batchSize)
        count2 = int(np.shape(memory2)[0] / otherSize)

        newMemory1 = memory1
        newMemory2 = memory2

        for i in range(epoch):
            n_examples = np.shape(newMemory1)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            newMemory1 = newMemory1[index2]

            n_examples = np.shape(newMemory2)[0]
            index3 = [i for i in range(n_examples)]
            np.random.shuffle(index3)
            newMemory2 = newMemory2[index3]

            for j in range(count):
                batchX = newMemory1[j * batchSize:(j + 1) * batchSize]

                c1 = j % count2
                batchX2 = newMemory2[c1 * otherSize:(c1 + 1) * otherSize]

                batchX = torch.cat([batchX,batchX2],0)
                self.vae.Update(batchX)
                '''
                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()
                '''

    def Train_Self_(self, epoch, dataX,teacher):
        batchsize = 128
        model = self.vae

        batchSize = self.batch_size
        batchSize = int(batchSize / teacher.GetTeacherCount())
        otherSize = self.batch_size - batchSize * (teacher.GetTeacherCount()-1)

        count = int(np.shape(dataX)[0] / otherSize)
        tcount = int(np.shape(teacher.teacherArray[0].currentKnowledge)[0] / batchSize)

        for i in range(epoch):
            n_examples = np.shape(dataX)[0]
            index2 = [i for i in range(n_examples)]
            np.random.shuffle(index2)
            dataX = dataX[index2]

            for j in range(count):
                batchX = dataX[j * otherSize:(j + 1) * otherSize]

                c1 = j % tcount

                for t1 in range(teacher.GetTeacherCount() - 1):
                    a1 = teacher.teacherArray[t1].currentKnowledge[c1 * batchSize:(c1 + 1) * batchSize]
                    batchX = torch.cat([batchX, a1], 0)

                self.optimizer.zero_grad()
                loss = self.training_step(batchX)
                loss.backward()
                self.optimizer.step()

    def Give_MeanAndVar(self,tensor):
        return self.vae.Give_MeanAndVar(tensor)
