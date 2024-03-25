import torch
from models import BaseVAE
from models import nn
from torch.nn import functional as F
from .types_ import *
import numpy as np


class FullVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 beta: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(FullVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.beta = beta

        modules = []
        self.label1 = nn.Sequential(
            nn.Linear(input_size*input_size*3, 2000),
                        nn.LeakyReLU())
        modules.append(self.label1)

        self.label2 = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.LeakyReLU())
        modules.append(self.label2)

        self.label3 = nn.Sequential(
            nn.Linear(1000, 800),
            nn.LeakyReLU())
        modules.append(self.label3)

        self.label4 = nn.Sequential(
            nn.Linear(800, 500),
            nn.LeakyReLU())
        modules.append(self.label4)
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(500, latent_dim)
        self.fc_var = nn.Linear(500, latent_dim)

        #decoder
        modules = []
        self.decoder_label1 = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.LeakyReLU())
        modules.append(self.decoder_label1)

        self.decoder_label2 = nn.Sequential(
            nn.Linear(500, 800),
            nn.LeakyReLU())
        modules.append(self.decoder_label2)

        self.decoder_label3 = nn.Sequential(
            nn.Linear(800, 1000),
            nn.LeakyReLU())
        modules.append(self.decoder_label3)

        self.decoder_label4 = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.LeakyReLU())
        modules.append(self.decoder_label4)
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(2000, input_size * input_size * 3),
            nn.Tanh())

    def GiveFeatures(self,x):
        with torch.no_grad():
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu,log_var)
        return z

    def training_step(self,batch):
        results = self.forward(batch)

        train_loss = self.loss_function(*results)

        return train_loss['loss']

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        input = input.view(-1,self.input_size*self.input_size*3)
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        #result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(z)
        result = self.final_layer(result)
        result = result.view(-1,3,self.input_size,self.input_size)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def Give_MeanAndVar(self,tensor):
        return self.encode(tensor)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 1.0
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, input, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def Give_FullSampleByLatent(self,latentSample):

        with torch.no_grad():
            arr = []
            batchsize = 64
            count = int(np.shape(latentSample)[0] / batchsize)
            for i in range(count):
                latentBatch = latentSample[i*batchsize:(i+1)*batchsize]
                resultBatch = self.decode(latentBatch)
                if np.shape(arr)[0] == 0:
                    arr = resultBatch
                else:
                    arr = torch.cat([arr,resultBatch],0)
            return arr

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]