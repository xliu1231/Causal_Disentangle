import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


# https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide

# Environment variables set by torch.distributed.launch

# LOCAL_RANK defines the ID of a worker within a node
# If we have 2 nodes and 2 workers/node then 0,1, 0,1
#LOCAL_RANK = int(os.environ['LOCAL_RANK'])  
# WORLD_SIZE defines the total number of workers. 
# We have 2 nodes and 2 workers/node then WORLD_SIZE=4
#WORLD_SIZE = int(os.environ['WORLD_SIZE'])
# 0, 1, 2, 3
#WORLD_RANK = int(os.environ['RANK'])



def setup(rank, world_size):
    
    # initialize the process group
               
    print('Initializing worker_{}/{}\n'.format(rank, world_size))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def main():

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    #model = ToyModel().to(rank)
    model = ToyModel().to(rank)
    print("model is on ", next(model.parameters()).device)
    #ddp_model = DDP(model, device_ids=[rank])
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()

    outputs = ddp_model(torch.ones(200, 10))
    labels = torch.randn(200, 5).to(rank)

    loss = loss_fn(outputs, labels)
    print("Loss is ",loss.item())

    loss.backward()
    optimizer.step()

    cleanup()
    


if __name__ == '__main__':
    
    main()
  

#python3 -m torch.distributed.launch \
#--nproc_per_node=2 --nnodes=2 --node_rank=0 \
#--master_addr=104.171.200.62 --master_port=1234 \
#multigpu_example.py
