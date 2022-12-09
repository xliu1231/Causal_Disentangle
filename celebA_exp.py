"""

    This is the code template for running experiments on celebA dataset
    

"""

import argparse
import os


import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from vae import VAE

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")



class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, data_path, attr_path, attr=[], transform=None):
    
        df = pd.read_csv(attr_path, sep="\s+", skiprows=1, index_col=0)
        df = df.replace(-1, 0)
        self.data_path = data_path
        self.attr_path = attr_path
        self.img_names = df.index.values
        self.target = df[attr].values if attr else df.values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.target[index]
        return img, label

    def __len__(self):
        return self.target.shape[0]



def main():
    
    global args
    
    args = parse()
    
    
    # set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if(args.local_rank == 0):
        print("model: {}".format(args.model))
        print("dataset: {}".format(args.dataset))
        print("batch size is {}".format(args.batch_size))
        
    cudnn.benchmark = True # for amp setting
    
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        

    memory_format = torch.contiguous_format
        
    # load and set models
    model = VAE(z_dim=args.z_dim)
    model = model.cuda().to(memory_format=memory_format)
    
    if args.distributed:
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)
    
    # Scale learning rate based on global batch size
    batch_size  = args.batch_size

    args.learning_rate = args.learning_rate*float(args.batch_size*args.world_size)/256.

    # optimizer and corresponding scheduler
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
         optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
            
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()
            
    
    ## Paths    
    # output path and model path
    if not os.path.exists(args.outputs_path):
         raise Exception("outputs path not created.")
            
    if not os.path.exists(args.model_path):
         raise Exception("model path not created.")
            
      
    ## Data Processing
    # number of worker for dataloaders
    num_workers = 4
    
    image_size = 64
    transform = T.Compose([
        T.Resize([image_size, image_size]),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CelebaDataset(data_path=args.data_path,
                            attr_path=args.attr_path,
                            attr=args.attr_list,
                            transform=transform)
    
    train_sampler = None

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    train_samples = len(train_loader) * batch_size
    args.train_samples = train_samples

    if(args.local_rank == 0):
        print("# of training samples: ", train_samples)
        
    train_time = 0.0
    for epoch in range(args.start_epoch, args.epoch_num):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch)
        
        
        # save checkpoint
        if args.local_rank == 0 and epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, str(args.model) + "_" + str(epoch))
            
            
def train(train_loader, model, optimizer, epoch):
    """
        Model training

    """
    batch_time = AverageMeter()
    losses = AverageMeter()
   

    # switch to train mode
    model.train()
    end = time.time()
    total_start = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # compute output
        output = model(input)
        
        recons_loss = F.mse_loss(output[0], input)
       
        mu = output[1]
        log_var = output[2]
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_weight = input.shape[0] / args.train_samples
        
        loss = recons_loss + kld_loss * kld_weight

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()


            if args.local_rank == 0:
                learning_rate = optimizer.param_groups[0]['lr']
    
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.5f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader),
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses))
       
        input, target = prefetcher.next()
        
    torch.cuda.synchronize()
    if args.local_rank == 0:
        print("[Training] Epoch:{0} total_time: {1:.3f}".format(
            epoch, time.time()-total_start))
        
        
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


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.learning_rate*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

    
def save_checkpoint(state, filename):
    filename += "_checkpoint.pth.tar"
    filename = args.model_path +'/'+ filename
    torch.save(state, filename)
    
    
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def parse():
    parser = argparse.ArgumentParser(description =
        "causal disentanglement on CelebA dataset.")
    
    
    ## Data format (Pytorch format)
    # batch size (0) x channels (1) x height (2) x width (3)
    parser.add_argument("--batch-size",  default = 128,  type = int, metavar='B',
        help = "The batch size for training.")
    
    
    # image format: channels (1), height (2), width (3)
    parser.add_argument("--image-height", default =  64, type = int,
        help = "The image height of each sample.")
    parser.add_argument("--image-width",  default =  64, type = int,
        help = "The image width  of each sample.")
    parser.add_argument("--image-channels", default = 3, type = int,
        help = "The number of channels in each sample.")
    
   
    
    
    ## Paths (Data, Checkpoints and Results)
    # inputs: data
    parser.add_argument("--dataset", default="CelebA", type=str, help="The dataset used for training.")
    parser.add_argument("--data-path", default= "/fs/cml-datasets/CelebA/Img/img_align_celeba", type=str,
                    help="The path to the folder stroing the data.")
    parser.add_argument("--attr-path", default= "/fs/cml-datasets/CelebA/Anno/list_attr_celeba.txt", type=str,
                    help="The path to the folder stroing the attribute of the data.")
    parser.add_argument("--attr-list", default= ['Eyeglasses', 'Smiling', 'Wearing_Hat', 'Wearing_Earrings'], type=list,
                    help="The path to the folder stroing the list of attributes of interest.")
    
    
    
    
    # outputs: checkpoints and statistics
    parser.add_argument("--outputs-path", default = "/nfshomes/xliu1231/Causal_Disentangle/outputs", type = str,
        help = "The path to the folder storing outputs from training.")
    parser.add_argument("--model-stamp", default = "", type = str,
        help = "The time stamp of the model (as a suffix to the its name).")
    parser.add_argument("--model-path", default = "/nfshomes/xliu1231/Causal_Disentangle/models", type = str,
        help = "The folder for all checkpoints in training.")
    
    
    
    
    parser.add_argument("--model", default="vae", type=str,
        help = "The name of the model.")
    ## General Settings
    parser.add_argument("--seed", type=int, default=1, metavar='S', help="random seed (default: 1)")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    
    
    ## Training Parameters
    # DDP
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument("--local_rank", default=0, type=int)
    
    #resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--z-dim', default=16, type=int)
    parser.add_argument('--epoch-num', default=100, type=int)
    parser.add_argument("--train-ratio", default = 0.9, type=float,
        help="The ratio of training samples in the .")
    parser.add_argument("--print_freq", default = 10, type = int,
        help = "Log the learning curve every print_freq iterations.")


    # learning rate scheduling
    parser.add_argument("--learning-rate", default = 0.3, type = float,
        help = "Initial learning rate of the optimizer.")

    parser.add_argument("--learning-rate-decay", dest = "rate_decay", action = 'store_true',
        help = "Learning rate is decayed during training.")
    parser.add_argument("--learning-rate-fixed", dest = "rate_decay", action = 'store_false',
        help = "Learning rate is fixed during training.")
    parser.set_defaults(rate_decay = True)

    # if rate_decay is True (--learning-rate-decay)
    parser.add_argument("--decay-epoch", default = "30", type = str,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")
    parser.add_argument("--decay-rate", default = 0.5, type = float,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.set_defaults(sgd=True)
    
    return parser.parse_args()





if __name__ == "__main__":
    main()
