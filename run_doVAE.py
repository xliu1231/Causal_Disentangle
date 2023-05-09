import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import datasets
from torchvision import transforms as T
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn import functional as F

sys.path.insert(0,"/nfshomes/xliu1231/Causal_Disentangle/3dshape_dataset")

from datasets import ShapeDataset,CelebaDataset
from dataset_3d_shape import sample_3dshape_dataset

rank = int(os.environ['RANK'])
device = torch.device("cuda:{}".format(rank))

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    

    
# Everytime you want to update your model, put it here.
# 
# run the following command in the command line 
#
# python -m torch.distributed.launch --nproc_per_node=4 run_doVAE.py 
# 
#     
class doVAE(nn.Module):

    def __init__(self,z_dim, c_dim = 2, in_channels=3):
        super(doVAE, self).__init__()
        
        self.z_dim = z_dim
        self.c_dim = c_dim
        # encoder output dim: c_Num*z_dim for mu_z, c_Num*z_dim for sigma_z, c_Num for mu_pi, c_Num for sigma_mu
        self.model_encoder= nn.Sequential(
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
            nn.Linear(256,  c_dim*(2*z_dim+2)),      
        )
        
            
        self.model_decoder = nn.Sequential(
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
        
                
    def encode(self, X):
        
        result = self.model_encoder(X)
        X_len = X.shape[0]
        mu = torch.zeros(X_len,self.c_dim * self.z_dim)
        logvar = torch.zeros(X_len,self.c_dim * self.z_dim)
        mu_pi = torch.zeros(X_len,self.c_dim)
        var_pi = torch.zeros(X_len,self.c_dim)
        for c in range(self.c_dim):
            mu[:,c*self.z_dim:(c+1)*self.z_dim] = result[:,c*self.z_dim : (c+1)*self.z_dim]
            logvar[:,c*self.z_dim:(c+1)*self.z_dim] = result[:,(self.c_dim+c)*self.z_dim : (self.c_dim+c+1)*self.z_dim]
            mu_pi[:,c] = result[:,2*self.c_dim*self.z_dim + c]
            var_pi[:,c] = result[:,2*self.c_dim*self.z_dim + self.c_dim + c]
                                
        
        return mu, logvar, mu_pi, var_pi
                                 
    
    def decode(self, z):
        
        return self.model_decoder(z)
    
    
    def generate(self, num_samples, current_device):                         
        z = torch.randn(num_samples, self.z_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples
    

    def reparameterize(self, mu, logvar,mu_pi,var_pi):
        X_len = mu.shape[0]
        z_all = torch.zeros(X_len, self.z_dim)
        Pi = torch.zeros(X_len, self.c_dim)
        for c in range(self.c_dim):
            std = torch.exp(0.5 * logvar[:,c*self.z_dim:(c+1)*self.z_dim])
            eps = torch.randn_like(std)
            curr_z = eps * std + mu[:,c*self.z_dim:(c+1)*self.z_dim]

            std_pi = torch.exp(0.5 * var_pi[:,c])
            eps_pi = torch.randn_like(std_pi)
            curr_pi = eps_pi * std_pi + mu_pi[:,c]
            curr_pi = curr_pi

            z_all += curr_z * curr_pi[:,None]
            Pi[:,c] = curr_pi
        return z_all, Pi
    
    

    def forward(self, X):
        mu, logvar, mu_pi, var_pi = self.encode(X)
        z,Pi = self.reparameterize(mu, logvar, mu_pi, var_pi)
        z,Pi = z.to(device), Pi.to(device)
        reconstructed_z = self.decode(z)
        return reconstructed_z, mu, logvar, Pi
    
def calculate_loss(X, reconstructed_z, mu, logvar, Pi, label, pi_weight = 1.0):
    """
    Loss term currently contains: reconstruction loss, classification loss for p(c|x)
    """
    # classification loss:
    criterion = nn.CrossEntropyLoss()
    pi_loss = criterion(Pi, label.float())
    
    # reconstruction loss:
    rec_loss = F.mse_loss(reconstructed_z, X)
    
    loss = rec_loss + pi_weight * pi_loss
    
    return loss, rec_loss, pi_loss
        
    

################ End of Model update

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    data_path = "/fs/cml-datasets/CelebA/Img/img_align_celeba"
    attr_path =  "/fs/cml-datasets/CelebA/Anno/list_attr_celeba.txt"

    image_size = 64
    transform = T.Compose([
        T.Resize([image_size, image_size]),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    model = doVAE(z_dim=4)
    device = torch.device("cuda:{}".format(rank))
    model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    train_set = CelebaDataset(data_path,attr_path,
                              attr=['Eyeglasses', 'Smiling'],
                              transform=transform)
    
    train_sampler = DistributedSampler(dataset=train_set)
    train_loader = DataLoader(dataset=train_set, 
                              batch_size=128, 
                              sampler=train_sampler,
                              num_workers=4)
    
    learning_rate = 0.01*float(128)/256.
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, 
                                momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(15):
        ddp_model.train()
        i = 0
        for x, target in train_loader:
            # randomly generate target set, delete afterward
            x = x.to(device)
            summ = torch.sum(target, dim = 1)
            zero_index = (summ == 0).nonzero().squeeze().detach()
            target[zero_index,:] = torch.tensor([1,0])
            two_index = (summ == 2).nonzero().squeeze().detach()
            target[two_index,:] = torch.tensor([0,1])
            target = target.to(device)

            i += 1
            batch_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            reconstructed_z, mu, logvar, Pi = ddp_model(x)
            loss, rec_loss, pi_loss = calculate_loss(x, reconstructed_z, mu, logvar, Pi, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%15 == 0 and rank==0:
        
                reduced_loss = loss.data
                losses.update(float(reduced_loss), x.size(0))
                torch.cuda.synchronize()
                batch_time.update((time.time() - end)/15)
                end = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.5f} ({loss.avg:.4f})\t'.format(
                            epoch, i, len(train_loader),
                            batch_time=batch_time,
                            loss=losses))

        
if __name__ == '__main__':
    
    main()        



