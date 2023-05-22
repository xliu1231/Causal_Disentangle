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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

sys.path.insert(0,"./3dshape_dataset")

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
class VAE(nn.Module):

    def __init__(self,z_dim, in_channels=3):
        super(VAE, self).__init__()
        
        self.z_dim = z_dim
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
            nn.Linear(256,  2*z_dim))
        
            
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
        
                
    def encode(self, x):
        result = self.model_encoder(x) # B x 2D
        mu = result[:, :self.z_dim]
        logvar = result[:, self.z_dim:]
        return mu, logvar
                                 
    
    def decode(self, z):
        
        return self.model_decoder(z)
    
    
    def reparametrize(self,mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        z = z.to(device)
        reconstructed_z = self.decode(z)
        return reconstructed_z, mu, logvar
    
    
    
def calculate_loss(X, reconstructed_z, mu, logvar, kl_weight = 1.0):
    # reconstruction loss:
    rec_loss = F.mse_loss(reconstructed_z, X)
    
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    
    loss = rec_loss +kl_weight * kld_loss
    
    return loss, rec_loss, kld_loss
        
    

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

    image_size = 64
    transform = T.Compose([
        T.Resize([image_size, image_size]),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    z_dim = 4
    model = VAE(z_dim)
    device = torch.device("cuda:{}".format(rank))
    model = model.to(device)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    train_set = CelebaDataset(data_path,attr_path,
                              attr=['Eyeglasses'],
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
            # ### labels
            # enc = OneHotEncoder(handle_unknown='ignore')
            # enc.fit(target)
            # target = torch.from_numpy(enc.transform(target).toarray())
            # #####
            x = x.to(device)
            target = target.to(device)

            i += 1
            batch_time = AverageMeter()
            rec_losses = AverageMeter()
            kl_losses = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            reconstructed_z, mu, logvar = ddp_model(x)
            loss, rec_loss, kld_loss = calculate_loss(x, reconstructed_z, mu, logvar)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%15 == 0 and rank==0:
        
                reduced_loss = loss.data
                reduced_rec = rec_loss.data
                reduced_kl = kld_loss.data
                losses.update(float(reduced_loss), x.size(0))
                rec_losses.update(float(reduced_rec), x.size(0))
                kl_losses.update(float(reduced_kl), x.size(0))
                torch.cuda.synchronize()
                batch_time.update((time.time() - end)/15)
                end = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.5f} ({loss.avg:.4f})\t'
                      'Reconstruction loss {rec_loss.avg:.4f}\t'.format(
                            epoch, i, len(train_loader),
                            batch_time=batch_time,
                            loss=losses,
                            rec_loss = rec_losses))

        
if __name__ == '__main__':
    
    main()        



