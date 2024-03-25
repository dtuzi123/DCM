import torch
from models import BaseVAE
from models import nn
from torch.nn import functional as F
from .types_ import *
import numpy as np

class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size

        if self.input_size == 64:
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride = 2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                   hidden_dims[-1],
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                          kernel_size= 3, padding= 1),
                                nn.Tanh())
        else:#input size = 32
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()
            hidden_dims = hidden_dims[0:4]

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.Tanh(),
                nn.Conv2d(hidden_dims[-1], out_channels=3,
                          kernel_size=3, padding=1),
                nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

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
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

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

    def loss_function2(self,
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
        batch = args[4]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 1
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VanillaVAE_(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE_, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size

        if self.input_size == 64:
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride = 2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                   hidden_dims[-1],
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                          kernel_size= 3, padding= 1),
                                nn.Tanh())
        else:#input size = 32
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()
            hidden_dims = hidden_dims[0:4]

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.Tanh(),
                nn.Conv2d(hidden_dims[-1], out_channels=3,
                          kernel_size=3, padding=1),
                nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

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
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

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

    def Give_Features(self,tensor):
        result = self.encoder(tensor)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        mu = mu.cpu().detach().numpy()
        return mu

    def Give_MeanAndVar(self,tensor):
        return self.encode(tensor)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function2(self,
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
        batch = args[4]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 0.01
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def loss_function_WithBeta(self,
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
        batch = args[4]
        beta = args[5]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = beta#0.01
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


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
        #kld_weight = 0.01
        kld_weight = 1
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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class Autoencoder_(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Autoencoder_, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size

        if self.input_size == 64:
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride = 2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                   hidden_dims[-1],
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                          kernel_size= 3, padding= 1),
                                nn.Tanh())
        else:#input size = 32
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            #self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()
            hidden_dims = hidden_dims[0:4]

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.Tanh(),
                nn.Conv2d(hidden_dims[-1], out_channels=3,
                          kernel_size=3, padding=1),
                nn.Tanh())

    def GiveCode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        with torch.no_grad():
            result = self.encoder(input)
            result = torch.flatten(result, start_dim=1)

            # Split the result into mu and var components
            # of the latent Gaussian distribution
            mu = self.fc_mu(result)
            #log_var = self.fc_var(result)
            mu = mu.view(np.shape(input)[0],-1)

            #return [mu, log_var]
            return mu

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        #log_var = self.fc_var(result)

        #return [mu, log_var]
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

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

    def Give_Features(self,tensor):
        result = self.encoder(tensor)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        mu = mu.cpu().detach().numpy()
        return mu

    def Give_MeanAndVar(self,tensor):
        return self.encode(tensor)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu = self.encode(input)
        z = mu
        return  [self.decode(z), input]

    def loss_function2(self,
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
        batch = args[4]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 0.01
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def loss_function_WithBeta(self,
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
        batch = args[4]
        beta = args[5]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = beta#0.01
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def CalculateMSE2(self,a,b):
        recons_loss = F.mse_loss(a, b, size_average=True)
        recons_loss = recons_loss.cpu().detach().numpy()
        return recons_loss

    def CalculateMse(self,input1,input2):
        arr = []
        for i in range(np.shape(input1)[0]):
            dis = self.CalculateMSE2(input1[i],input2[i])
            arr.append(dis)
        return arr


    def GiveLoss(self,dataSet):
        """
                Computes the VAE loss function.
                KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
                :param args:
                :param kwargs:
                :return:
                """
        input = dataSet
        reco = self.forward(input)
        reco = reco[0]

        # mu = args[2]
        # log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        # kld_weight = 0.01
        kld_weight = 1
        # recons_loss =F.mse_loss(recons, input)
        #recons_loss = F.mse_loss(reco, input, size_average=False,reduction="none")
        #loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        recons_loss = self.CalculateMse(input,reco)
        return recons_loss

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
        #mu = args[2]
        #log_var = args[3]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        #kld_weight = 0.01
        kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, input, size_average=False) / input.size(0)

        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss #+ kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach()}

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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class Autoencoder196_(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Autoencoder196_, self).__init__()

        latent_dim = 192
        self.latent_dim = latent_dim
        self.input_size = input_size

        if self.input_size == 64:
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride = 2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                   hidden_dims[-1],
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                          kernel_size= 3, padding= 1),
                                nn.Tanh())
        else:#input size = 32
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            if hidden_dims is None:
                hidden_dims = [64, 128, 256, 512, 1024]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            #self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()
            hidden_dims = hidden_dims[0:4]

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.Tanh(),
                nn.Conv2d(hidden_dims[-1], out_channels=3,
                          kernel_size=3, padding=1),
                nn.Tanh())

    def GiveCode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        #log_var = self.fc_var(result)
        #mu = mu.view(np.shape(input)[0],-1)
        mu = mu.view(np.shape(input)[0],3,8,8)

        #return [mu, log_var]
        return mu

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        #log_var = self.fc_var(result)

        #return [mu, log_var]
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

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

    def Give_Features(self,tensor):
        result = self.encoder(tensor)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        mu = mu.cpu().detach().numpy()
        return mu

    def Give_MeanAndVar(self,tensor):
        return self.encode(tensor)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu = self.encode(input)
        z = mu
        return  [self.decode(z), input]

    def loss_function2(self,
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
        batch = args[4]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 0#0.01
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def loss_function_WithBeta(self,
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
        batch = args[4]
        beta = args[5]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 0#beta#0.01
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def CalculateMSE2(self,a,b):
        recons_loss = F.mse_loss(a, b, size_average=True)
        recons_loss = recons_loss.cpu().detach().numpy()
        return recons_loss

    def CalculateMse(self,input1,input2):
        arr = []
        for i in range(np.shape(input1)[0]):
            dis = self.CalculateMSE2(input1[i],input2[i])
            arr.append(dis)
        return arr


    def GiveLoss(self,dataSet):
        """
                Computes the VAE loss function.
                KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
                :param args:
                :param kwargs:
                :return:
                """
        input = dataSet
        reco = self.forward(input)
        reco = reco[0]

        # mu = args[2]
        # log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        # kld_weight = 0.01
        kld_weight = 1
        # recons_loss =F.mse_loss(recons, input)
        #recons_loss = F.mse_loss(reco, input, size_average=False,reduction="none")
        #loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        recons_loss = self.CalculateMse(input,reco)
        return recons_loss

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
        #mu = args[2]
        #log_var = args[3]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        #kld_weight = 0.01
        kld_weight = 1.0
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, input, size_average=False) / input.size(0)

        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss #+ kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach()}

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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



class VAE196_(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE196_, self).__init__()

        latent_dim = 192
        self.latent_dim = latent_dim
        self.input_size = input_size

        if self.input_size == 64:
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride = 2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                   hidden_dims[-1],
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                          kernel_size= 3, padding= 1),
                                nn.Tanh())
        else:#input size = 32
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            #self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_mu = nn.Sequential(
                nn.Linear(hidden_dims[-1], latent_dim),
                nn.Tanh())
            self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()
            hidden_dims = hidden_dims[0:4]

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.Tanh(),
                nn.Conv2d(hidden_dims[-1], out_channels=3,
                          kernel_size=3, padding=1),
                nn.Tanh())

    def GiveCode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        #mu = mu.view(np.shape(input)[0],-1)
        #z = self.reparameterize(mu,log_var)
        z = mu
        z = z.view(np.shape(input)[0],3,8,8)

        #return [mu, log_var]
        return z

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu,log_var
        #z = self.reparameterize(mu,log_var)
        #return z
        #return [mu, log_var]
        #return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

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

    def Give_Features(self,tensor):
        result = self.encoder(tensor)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        mu = mu.cpu().detach().numpy()
        return mu

    def Give_MeanAndVar(self,tensor):
        return self.encode(tensor)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z = mu
        return  [self.decode(z), input, mu, log_var]


    def loss_function2(self,
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
        batch = args[4]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 0.01
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss #+ kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def loss_function_WithBeta(self,
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
        batch = args[4]
        beta = args[5]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = beta#0.01
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss #+ kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}


    def CalculateMSE2(self,a,b):
        recons_loss = F.mse_loss(a, b, size_average=True)
        recons_loss = recons_loss.cpu().detach().numpy()
        return recons_loss

    def CalculateMse(self,input1,input2):
        arr = []
        for i in range(np.shape(input1)[0]):
            dis = self.CalculateMSE2(input1[i],input2[i])
            arr.append(dis)
        return arr


    def GiveLoss(self,dataSet):
        """
                Computes the VAE loss function.
                KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
                :param args:
                :param kwargs:
                :return:
                """
        input = dataSet
        reco = self.forward(input)
        reco = reco[0]

        # mu = args[2]
        # log_var = args[3]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        # kld_weight = 0.01
        kld_weight = 1
        # recons_loss =F.mse_loss(recons, input)
        #recons_loss = F.mse_loss(reco, input, size_average=False,reduction="none")
        #loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        recons_loss = self.CalculateMse(input,reco)
        return recons_loss

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
        batch = input#args[4]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 0.2
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss #+ kld_weight * kld_loss
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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



class VanillaVAE_Flexible(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size: int,
                 beta: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE_Flexible, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.beta = beta

        if self.input_size == 64:
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size= 3, stride= 2, padding  = 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride = 2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                   hidden_dims[-1],
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                          kernel_size= 3, padding= 1),
                                nn.Tanh())
        else:#input size = 32
            modules = []
            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=h_dim,
                                  kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU())
                )
                in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            '''
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
            '''

            self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()
            hidden_dims = hidden_dims[0:4]

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )

            self.decoder = nn.Sequential(*modules)

            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   hidden_dims[-1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.Tanh(),
                nn.Conv2d(hidden_dims[-1], out_channels=3,
                          kernel_size=3, padding=1),
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


    def LoadCACDFromPath(self,path):
        batch = [GetImage_cv(
            sample_file,
            input_height=256,
            input_width=256,
            resize_height=self.input_size,
            resize_width=self.input_size,
            crop=False)
            for sample_file in path]
        return batch

    def Train_Cpu_WithFiles(epoch, memoryBuffer):
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

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

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
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

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
        kld_weight = self.beta
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, input, size_average=False) / input.size(0)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def loss_function2(self,
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
        batch = args[4]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = self.beta
        #kld_weight = 1
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = F.mse_loss(recons, batch, size_average=False) / input.size(0)

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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
