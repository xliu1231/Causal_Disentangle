from abc import abstractmethod

import time
import argparse
import os


import torch
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F

import pandas as pd
import numpy as np
import einops

from vae import BaseVAE


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p)
    and sum over the last dimension
    
    Return: kl between each sample
    """
    # element-wise operation
    #qm, qv, pm, pv = qm.to(device), qv.to(device), pm.to(device), pv.to(device)
    kl =  0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    # sum over all dimensions except for batch
    kl = kl.sum(-1)
    kl = kl.sum(-1)
    # print("log var1", qv)
    if torch.isnan(kl.any()):
        print("\n\n\n\noveflow\n\n\n\n\n\n")
    return kl

def conditional_sample_gaussian(m,v, device):
    sample = torch.randn(m.size()).to(device)
    z = m.to(device) + (v.to(device)**0.5)*sample
    return z
    


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    
    
class CausalDAG(nn.Module):
    """
    creates a causal diagram A
    
    
    """
    def __init__(self, num_concepts, dim_per_concept, inference = False, bias=False, g_dim=32):
        
        super(CausalDAG, self).__init__()
        self.num_concepts = num_concepts
        self.dim_per_concept = dim_per_concept
        
        self.A = nn.Parameter(torch.zeros(num_concepts, num_concepts))
        self.I = nn.Parameter(torch.eye(num_concepts))
        self.I.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(num_concepts))
        else:
            self.register_parameter('bias', None)
            
        nets_z = []
        nets_label = []
        
        
        for _ in range(num_concepts):
            nets_z.append(
                nn.Sequential(
                    nn.Linear(dim_per_concept, g_dim),
                    nn.ELU(),
                    nn.Linear(g_dim, dim_per_concept)
                )
            )
                
            nets_label.append(
                nn.Sequential(
                    nn.Linear(1, g_dim),
                    nn.ELU(),
                    nn.Linear(g_dim, 1)
                )
            )
        self.nets_z = nn.ModuleList(nets_z)
        self.nets_label = nn.ModuleList(nets_label)
        
            
    def calculate_z(self, epsilon):
        """
        convert epsilon to z using the SCM assumption and causal diagram A
        
        """
        
        C = torch.inverse(self.I - self.A.t())
            
        if epsilon.dim() > 2: # one concept is represented by multiple dimensions     
            z = F.linear(epsilon.permute(0,2,1), C, self.bias)
            z = z.permute(0,2,1).contiguous() 
            
        else:
            z = F.linear(epsilon, C, self.bias)
        return z
    
    def calculate_epsilon(self, z):
        """
        convert epsilon to z using the SCM assumption and causal diagram A
         
        """
        
        C_inv = self.I - self.A.t()
        
        if z.dim() > 2: # one concept is represented by multiple dimensions     
            epsilon = F.linear(z.permute(0,2,1), C_inv, self.bias)
            epsilon = epsilon.permute(0,2,1).contiguous() 
            
        else:
            epsilon = F.linear(z, C, self.bias)
        return epsilon
    
    def mask(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(dim=-1).cuda()
        res = torch.matmul(self.A.t(), x)
        return res
    
    def g_z(self, x):
        """
        apply nonlinearity for more stable approximation
        
        """
        x_flatterned = x.view(-1, self.num_concepts*self.dim_per_concept)
        concepts = torch.split(x_flatterned, self.dim_per_concept, dim = 1)
        res = []
        for i, concept in enumerate(concepts):
            t = self.nets_z[i](concept)
            res.append(t)
        x = torch.concat(res, dim=1).reshape([-1, self.num_concepts, self.dim_per_concept])
        return x
    
    def g_label(self, x):
        """
        apply nonlinearity for more stable approximation
        
        """
        x_flatterned = x.view(-1, self.num_concepts)
        concepts = torch.split(x_flatterned, 1, dim = 1)
        res = []
        for i, concept in enumerate(concepts):
            res.append(self.nets_label[i](concept))
        x = torch.concat(res, dim=1).reshape([-1, self.num_concepts])
        return x
            
    def forward(self, x):
        return self.g(self.mask(x))

    
class CausalVAE(BaseVAE):
    """
     causal VAE
     
    """
    def __init__(self,
                 z_dim,
                 num_concepts,
                 dim_per_concept,
                 in_channels=3,
                 lambdav = 0.001,
                 alpha = 1,
                 beta = 0.3,
                 gamma = 1,
                 ):
        super(CausalVAE, self).__init__()
        
        assert z_dim == num_concepts * dim_per_concept
        
        self.z_dim = z_dim
        self.num_concepts = num_concepts
        self.dim_per_concept = dim_per_concept
        self.lambdav = lambdav
        
        # network modules
        self.dag = CausalDAG(num_concepts, dim_per_concept)
        # attention weight
        self.W = nn.Parameter(torch.nn.init.normal_(torch.zeros(num_concepts, num_concepts), mean=0, std=1))
        
        # loss parameters
        self.alpha = alpha # reconstuction loss
        self.beta = beta # kl -> eps
        self.gamma = gamma # kl -> z 
        
        
        
        
        # potential input data B x 128 x 128
        
        # Encoder
        
        self.encoder = nn.Sequential(
            # B, 32, 64, 64
            nn.Conv2d(in_channels, out_channels=32, kernel_size=4, stride=2, padding=1),          
            nn.LeakyReLU(0.2, inplace=True),
            # B, 32, 32, 32
            nn.Conv2d(32, out_channels=32, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),
            # B, 64, 16, 16
            nn.Conv2d(32, out_channels=64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),
            # B,  64, 8, 8
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),
            # B,  64, 4,  4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),          
            nn.LeakyReLU(0.2, inplace=True),
            # B, 256,  1,  1
            nn.Conv2d(64, 256, kernel_size=4, stride=2),           
            nn.LeakyReLU(0.2, inplace=True),
            View((-1, 256*1*1)),                 # B, 256
            # for mu and logvar
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        
        # Decoder
        
        self.decoder = nn.Sequential(
            View((-1, z_dim, 1, 1)),
            nn.Conv2d(z_dim, 128, 1),  
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4),      # B,  64,  4,  4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1)
        )
        

        self.weight_init()
        
    def weight_init(self):
        for block in self._modules:
            # skip initialization of dag module
            continue
            for m in self._modules[block]:
                kaiming_init(m)
                
    def encode(self, x):
        """
        Encodes the inputs by passing through the encoder
        :param x: (Tensor) Input tensor to encoder [B x C x H x W]        
        :return: (Tensor, Tensor) Mean and Log Variance vector of the Multi-Gaussian. [B x 2D]
        2D: D dimensional mu and D dimensional logvar
        
        The latent dimension has shape  B x C x D_c [batch x num_concepts x dim_per_concept]
        
        
        """
        result = self.encoder(x) # B x 2D
        e_mu = result[:, :self.z_dim]
        e_logvar = result[:, self.z_dim:]
        e_logvar = F.softplus(e_logvar) + 1e-8
        return e_mu, e_logvar
    
    
    
    def decode(self, z):
        """
        Maps the given latent representations to images
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)
    
    
    def generate(self, num_samples):
        """
        Samples [num_samples] latent vectors and maps them to images
        :param num_samples: (int) N number of samples to generate
        :param current_device: (int) device number, usually represented by variable device
        :return: (Tensor, Tensor) [N x C x H x W]
        
        """
        pass
    
    def generate_with_label(self, labels):
        """
        generate samples with certain labels
        
        """
        pass
          
    
    def attention(self, z, e):
        """
        calculate the attention score
        
        
        """
        attn = torch.matmul(self.W, z).matmul(e.permute(0,2,1))
        attn = torch.nn.Sigmoid()(attn)
        attn = torch.softmax(attn, dim = 1)
        e = torch.matmul(attn, e)
        
        return e
   
    
    def forward(self, x, label):
        """
        """
        B = x.size()[0] # batch size
        
        e_m, e_v = self.encode(x)
        latent_dim = [B, self.num_concepts, self.dim_per_concept]
        e_m, e_v = e_m.reshape(latent_dim), torch.ones(latent_dim).cuda()  
        
        # z = (I - A.T)^(-1) * eps 
        z_m, z_v = self.dag.calculate_z(e_m), torch.ones(latent_dim).cuda()  
        
        masked_z_m = self.dag.mask(z_m)
        masked_label = self.dag.mask(label)
        
        # apply nonlinearity
        masked_z_m = self.dag.g_z(masked_z_m)
        pred_label = self.dag.g_label(masked_label)
        
        # attention
        e_tilde = self.attention(e_m, z_m)
        
        # z for predicting label u
        z_approx = masked_z_m + e_tilde
        z = conditional_sample_gaussian(z_approx, e_v * self.lambdav, z_approx.device)
        
        rec_x = self.decode(z)
        
        
        # losses returns by number of samples in the batch
        
        # reconstruction loss
        rec = F.mse_loss(rec_x, x)
        
        # KL between eps ~ N(0,1) and Q_phi(eps|x,u)
        p_m, p_v = torch.zeros_like(e_m), torch.ones_like(e_v)
        kl = self.beta * kl_normal(e_m, e_v, p_m, p_v)
        
        # KL between Q_phi(z|x, u) and P_theta(z|u) 
        mean_label = label.mean(dim=0)
        max_label = label.max(dim=0).values
        normalized_label_mean = (label - mean_label) / max_label
        
        cp_m = einops.repeat(normalized_label_mean, 'b d -> b d repeat', repeat=self.dim_per_concept).cuda()  
        cp_v = torch.ones_like(z_v).cuda()  
        kl += self.gamma * kl_normal(z_m, z_v, cp_m, cp_v)
        
        if torch.isnan(kl.mean()):
            print(kl)
        
        kl = kl.mean()
        
        # constraints
        lm = kl_normal(z, cp_v, cp_m, cp_v)
        
        lm = lm.mean()
        
        lu = F.mse_loss(pred_label.squeeze(dim=-1).cuda(), label.cuda())
        lu = lu.mean()
        
            
        return rec, kl, lm, lu, rec_x, masked_label, self.dag.A
    