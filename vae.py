from abc import abstractmethod

import torch
from torch import nn
import torch.nn.init as init


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)



class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, x):
        """
        Encodes the inputs by passing through the encoder
        :param x: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) Mean and Log Variance vector of the Multi-Gaussian.
        """
        raise NotImplementedError

    def decode(self, z):
        """
        Maps the given latent representations to images
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        raise NotImplementedError

    def generate(self, num_samples, current_device, **kwargs):
        """
        Samples [num_samples] latent vectors and maps them to images
        :param num_samples: (int) N number of samples to generate
        :param current_device: (int) device number, usually represented by variable device
        :return: (Tensor, Tensor) [N x C x H x W]
        
        """
        raise NotImplementedError


    @abstractmethod
    def forward(self, *inputs):
        pass

    
class VAE(BaseVAE):
    """
     vanilla VAE
     
    """
    def __init__(self,
                 z_dim,
                 in_channels=3,
                 **kwargs):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        
        # Encoder
        
        self.encoder = nn.Sequential(
            # B,  32, 32, 32
            nn.Conv2d(in_channels, out_channels=32, kernel_size=4, stride=2, padding=1),          
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # B,  32, 16, 16
            nn.Conv2d(32, out_channels=32, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # B,  64,  8,  8
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # B,  64,  4,  4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),          
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # B, 256,  1,  1
            nn.Conv2d(64, 256, kernel_size=4, stride=1),           
            nn.BatchNorm2d(256),
            nn.ReLU(),
            View((-1, 256*1*1)),                 # B, 256
            
            # for mu and logvar
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        
        # Decoder
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # B, 3, 64, 64
        )

        self.weight_init()
        
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
                
    def encode(self, x):
        """
        Encodes the inputs by passing through the encoder
        :param x: (Tensor) Input tensor to encoder [B x C x H x W]
        :param z_dim: (int) dimension of the latent representation
        :return: (Tensor, Tensor) Mean and Log Variance vector of the Multi-Gaussian. [B x D]
        
        2D: D dimensional mu and D dimensional logvar
        """
        result = self.encoder(x) # B x 2D
        mu = result[:, :self.z_dim]
        logvar = result[:, self.z_dim:]
        return mu, logvar
    
    def decode(self, z):
        """
        Maps the given latent representations to images
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)
    
    
    def generate(self, num_samples, current_device):
        """
        Samples [num_samples] latent vectors and maps them to images
        :param num_samples: (int) N number of samples to generate
        :param current_device: (int) device number, usually represented by variable device
        :return: (Tensor, Tensor) [N x C x H x W]
        
        """
        z = torch.randn(num_samples, self.z_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples
        
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparametrize(mu, logvar)
        reconstructed_z = self.decode(z)
        return reconstructed_z, mu, logvar
    
    
    
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
            
def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu
    